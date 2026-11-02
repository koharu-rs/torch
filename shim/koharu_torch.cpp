#include "convert.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Context.h>
#include <ATen/DLConvertor.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/Parallel.h>
#include <ATen/Version.h>
#include <ATen/accelerator/Graph.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/Formatting.h>
#include <ATen/detail/CUDAHooksInterface.h>
#include <c10/core/Event.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/csrc/jit/serialization/pickle.h>

#include <sstream>
#include <string>
#include <unordered_map>

using namespace torch::aot_inductor;

namespace {

c10::DeviceType device_type(int32_t type) {
  return static_cast<c10::DeviceType>(type);
}

std::vector<at::Tensor> tensors(const AtenTensorHandle* handles, int64_t len) {
  return pointer_to_list<at::Tensor>(handles, len);
}

StringHandle new_string(const std::string& value) {
  StringHandle handle = nullptr;
  TORCH_CHECK(
      torch_new_string_handle(value.data(), value.size(), &handle) ==
          AOTI_TORCH_SUCCESS,
      torch_exception_get_what_without_backtrace());
  return handle;
}

void new_strings(
    const std::vector<std::string>& values,
    StringHandle** ret,
    int64_t* len) {
  *ret = aten_new_array<StringHandle>(values.size());
  *len = static_cast<int64_t>(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    (*ret)[i] = new_string(values[i]);
  }
}

c10::Stream stream(AtenStream stream) {
  return c10::Stream::unpack3(
      stream.id,
      static_cast<c10::DeviceIndex>(stream.device_index),
      device_type(stream.device_type));
}

AtenStream from_stream(const c10::Stream& stream) {
  return {
      stream.id(),
      stream.device_index(),
      static_cast<int32_t>(stream.device_type())};
}

AtenMemoryStat stat(const c10::CachingAllocator::StatArray& stats) {
  const auto& all =
      stats[static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE)];
  return {all.current, all.peak, all.allocated, all.freed};
}

at::Generator& generator(AtenGeneratorHandle handle) {
  return *generator_handle_to_generator_pointer(handle);
}

c10::InferenceMode* inference_mode(AtenInferenceModeHandle handle) {
  return reinterpret_cast<c10::InferenceMode*>(handle);
}

c10::Event* event(AtenEventHandle handle) {
  return reinterpret_cast<c10::Event*>(handle);
}

at::accelerator::Graph* graph(AtenGraphHandle handle) {
  return reinterpret_cast<at::accelerator::Graph*>(handle);
}

torch::inductor::AOTIModelPackageLoader* package(
    AtenAotiPackageHandle handle) {
  return reinterpret_cast<torch::inductor::AOTIModelPackageLoader*>(handle);
}

// Forwards the thread's warnings to a C callback.
class WarningHandler : public c10::WarningHandler {
 public:
  void process(const c10::Warning& warning) override {
    handler(warning.msg().c_str(), context);
  }

  AtenWarningHandler handler = nullptr;
  void* context = nullptr;
};

thread_local WarningHandler warning_handler;

} // namespace

void aten_free(void* array) {
  std::free(array);
}

// --- Autograd ---------------------------------------------------------------

AOTITorchError aten_create_inference_mode(
    AtenInferenceModeHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = reinterpret_cast<AtenInferenceModeHandle>(
        new c10::InferenceMode());
  });
}

AOTITorchError aten_delete_inference_mode(
    AtenInferenceModeHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete inference_mode(guard); });
}

AOTITorchError aten_requires_grad(AtenTensorHandle tensor, bool* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = tensor_handle_to_tensor_pointer(tensor)->requires_grad(); });
}

AOTITorchError aten_grad(AtenTensorHandle tensor, AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = new_tensor_handle(
        at::Tensor(tensor_handle_to_tensor_pointer(tensor)->grad()));
  });
}

AOTITorchError aten_set_grad(
    AtenTensorHandle tensor,
    AtenTensorHandle grad) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    tensor_handle_to_tensor_pointer(tensor)->mutable_grad() =
        grad ? *tensor_handle_to_tensor_pointer(grad) : at::Tensor();
  });
}

AOTITorchError aten_autograd_grad(
    const AtenTensorHandle* outputs,
    int64_t outputs_len,
    const AtenTensorHandle* inputs,
    int64_t inputs_len,
    const AtenTensorHandle* grad_outputs,
    int64_t grad_outputs_len,
    const bool* retain_graph,
    bool create_graph,
    bool allow_unused,
    AtenTensorHandle** ret,
    int64_t* ret_len) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto grads = torch::autograd::grad(
        tensors(outputs, outputs_len),
        tensors(inputs, inputs_len),
        grad_outputs ? tensors(grad_outputs, grad_outputs_len)
                     : std::vector<at::Tensor>(),
        retain_graph ? std::make_optional(*retain_graph) : std::nullopt,
        create_graph,
        allow_unused);
    aten_tensor_list(std::move(grads), ret, ret_len);
  });
}

// --- Random numbers ---------------------------------------------------------

AOTITorchError aten_manual_seed(uint64_t seed) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ at::manual_seed(seed); });
}

AOTITorchError aten_create_generator(
    int32_t type,
    int32_t index,
    AtenGeneratorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto made = device_type(type) == c10::DeviceType::CPU
        ? at::detail::createCPUGenerator()
        : at::globalContext()
              .getAcceleratorHooksInterface(device_type(type))
              .getNewGenerator(static_cast<c10::DeviceIndex>(index));
    *ret = generator_pointer_to_generator_handle(
        new at::Generator(std::move(made)));
  });
}

AOTITorchError aten_default_generator(
    int32_t type,
    int32_t index,
    AtenGeneratorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto& found = at::globalContext().defaultGenerator(
        c10::Device(device_type(type), static_cast<c10::DeviceIndex>(index)));
    *ret = generator_pointer_to_generator_handle(new at::Generator(found));
  });
}

AOTITorchError aten_generator_set_seed(
    AtenGeneratorHandle handle,
    uint64_t seed) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& gen = generator(handle);
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_current_seed(seed);
  });
}

AOTITorchError aten_generator_seed(
    AtenGeneratorHandle handle,
    uint64_t* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& gen = generator(handle);
    std::lock_guard<std::mutex> lock(gen.mutex());
    *ret = gen.current_seed();
  });
}

AOTITorchError aten_generator_set_offset(
    AtenGeneratorHandle handle,
    uint64_t offset) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& gen = generator(handle);
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_offset(offset);
  });
}

AOTITorchError aten_generator_offset(
    AtenGeneratorHandle handle,
    uint64_t* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& gen = generator(handle);
    std::lock_guard<std::mutex> lock(gen.mutex());
    *ret = gen.get_offset();
  });
}

AOTITorchError aten_generator_state(
    AtenGeneratorHandle handle,
    AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& gen = generator(handle);
    std::lock_guard<std::mutex> lock(gen.mutex());
    *ret = new_tensor_handle(gen.get_state());
  });
}

AOTITorchError aten_generator_set_state(
    AtenGeneratorHandle handle,
    AtenTensorHandle state) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& gen = generator(handle);
    std::lock_guard<std::mutex> lock(gen.mutex());
    gen.set_state(*tensor_handle_to_tensor_pointer(state));
  });
}

// --- Accelerator ------------------------------------------------------------

AOTITorchError aten_accelerator(int32_t* type, int32_t* count) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto accelerator = at::accelerator::getAccelerator(/*checked=*/false);
    *type = accelerator ? static_cast<int32_t>(*accelerator) : -1;
    *count = accelerator ? at::accelerator::deviceCount() : 0;
  });
}

AOTITorchError aten_set_device(int32_t index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::accelerator::setDeviceIndex(static_cast<c10::DeviceIndex>(index));
  });
}

AOTITorchError aten_synchronize(int32_t index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::accelerator::synchronizeDevice(static_cast<c10::DeviceIndex>(index));
  });
}

AOTITorchError aten_memory_stats(
    int32_t index,
    AtenMemoryStats* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto stats =
        at::accelerator::getDeviceStats(static_cast<c10::DeviceIndex>(index));
    *ret = {
        stat(stats.allocation),
        stat(stats.segment),
        stat(stats.active),
        stat(stats.inactive_split),
        stat(stats.allocated_bytes),
        stat(stats.reserved_bytes),
        stat(stats.active_bytes),
        stat(stats.inactive_split_bytes),
        stat(stats.requested_bytes),
        stats.num_alloc_retries,
        stats.num_ooms,
        stats.num_sync_all_streams,
        stats.num_device_alloc,
        stats.num_device_free,
    };
  });
}

AOTITorchError aten_reset_peak_memory_stats(int32_t index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::accelerator::resetPeakStats(static_cast<c10::DeviceIndex>(index));
  });
}

AOTITorchError aten_reset_accumulated_memory_stats(int32_t index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::accelerator::resetAccumulatedStats(
        static_cast<c10::DeviceIndex>(index));
  });
}

AOTITorchError aten_memory_info(
    int32_t index,
    uint64_t* free,
    uint64_t* total) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto [device_free, device_total] =
        at::accelerator::getMemoryInfo(static_cast<c10::DeviceIndex>(index));
    *free = device_free;
    *total = device_total;
  });
}

AOTITorchError aten_empty_cache(void) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { at::accelerator::emptyCache(); });
}

AOTITorchError aten_empty_host_cache(void) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { at::accelerator::emptyHostCache(); });
}

// --- Streams and events -----------------------------------------------------

AOTITorchError aten_current_stream(
    int32_t index,
    AtenStream* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = from_stream(at::accelerator::getCurrentStream(
        static_cast<c10::DeviceIndex>(index)));
  });
}

AOTITorchError aten_set_current_stream(AtenStream value) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { at::accelerator::setCurrentStream(stream(value)); });
}

AOTITorchError aten_stream_from_pool(
    int32_t type,
    int32_t index,
    bool high_priority,
    AtenStream* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = from_stream(
        c10::impl::getDeviceGuardImpl(device_type(type))
            ->getStreamFromGlobalPool(
                c10::Device(
                    device_type(type), static_cast<c10::DeviceIndex>(index)),
                high_priority));
  });
}

AOTITorchError aten_stream_synchronize(AtenStream value) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { stream(value).synchronize(); });
}

AOTITorchError aten_stream_query(AtenStream value, bool* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = stream(value).query(); });
}

AOTITorchError aten_create_event(
    int32_t type,
    bool enable_timing,
    AtenEventHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    // A backend's default event records timing; PyTorch's default doesn't.
    *ret = reinterpret_cast<AtenEventHandle>(new c10::Event(
        device_type(type),
        enable_timing ? c10::EventFlag::BACKEND_DEFAULT
                      : c10::EventFlag::PYTORCH_DEFAULT));
  });
}

AOTITorchError aten_delete_event(AtenEventHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ delete event(handle); });
}

AOTITorchError aten_event_record(
    AtenEventHandle handle,
    AtenStream value) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { event(handle)->record(stream(value)); });
}

AOTITorchError aten_event_block(
    AtenEventHandle handle,
    AtenStream value) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { event(handle)->block(stream(value)); });
}

AOTITorchError aten_event_query(AtenEventHandle handle, bool* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = event(handle)->query(); });
}

AOTITorchError aten_event_synchronize(AtenEventHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { event(handle)->synchronize(); });
}

AOTITorchError aten_event_elapsed_time(
    AtenEventHandle start,
    AtenEventHandle end,
    double* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = event(start)->elapsedTime(*event(end)); });
}

// --- Graphs -----------------------------------------------------------------

AOTITorchError aten_create_graph(
    bool keep_graph,
    AtenGraphHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = reinterpret_cast<AtenGraphHandle>(
        new at::accelerator::Graph(keep_graph));
  });
}

AOTITorchError aten_delete_graph(AtenGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ delete graph(handle); });
}

AOTITorchError aten_graph_capture_begin(
    AtenGraphHandle handle,
    AtenMempoolId pool,
    AtenGraphCaptureMode mode) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    graph(handle)->capture_begin(
        {pool.first, pool.second}, static_cast<at::GraphCaptureMode>(mode));
  });
}

AOTITorchError aten_graph_capture_end(AtenGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ graph(handle)->capture_end(); });
}

AOTITorchError aten_graph_instantiate(AtenGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ graph(handle)->instantiate(); });
}

AOTITorchError aten_graph_replay(AtenGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ graph(handle)->replay(); });
}

AOTITorchError aten_graph_reset(AtenGraphHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ graph(handle)->reset(); });
}

AOTITorchError aten_graph_pool(
    AtenGraphHandle handle,
    AtenMempoolId* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto [first, second] = graph(handle)->pool();
    *ret = {first, second};
  });
}

// --- Autocast ---------------------------------------------------------------

AOTITorchError aten_set_autocast_enabled(int32_t type, bool enabled) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { at::autocast::set_autocast_enabled(device_type(type), enabled); });
}

AOTITorchError aten_autocast_enabled(int32_t type, bool* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = at::autocast::is_autocast_enabled(device_type(type)); });
}

AOTITorchError aten_set_autocast_dtype(int32_t type, int32_t dtype) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::autocast::set_autocast_dtype(
        device_type(type), static_cast<at::ScalarType>(dtype));
  });
}

AOTITorchError aten_autocast_dtype(int32_t type, int32_t* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = static_cast<int32_t>(
        at::autocast::get_autocast_dtype(device_type(type)));
  });
}

AOTITorchError aten_set_autocast_cache_enabled(bool enabled) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { at::autocast::set_autocast_cache_enabled(enabled); });
}

AOTITorchError aten_autocast_cache_enabled(bool* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = at::autocast::is_autocast_cache_enabled(); });
}

AOTITorchError aten_autocast_increment_nesting(int32_t* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = at::autocast::increment_nesting(); });
}

AOTITorchError aten_autocast_decrement_nesting(int32_t* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = at::autocast::decrement_nesting(); });
}

AOTITorchError aten_autocast_clear_cache(void) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { at::autocast::clear_cache(); });
}

// --- Settings ---------------------------------------------------------------

AOTITorchError aten_set_setting(AtenSetting setting, int64_t value) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& context = at::globalContext();
    const bool on = value != 0;
    switch (setting) {
      case ATEN_NUM_THREADS:
        at::set_num_threads(static_cast<int>(value));
        break;
      case ATEN_NUM_INTEROP_THREADS:
        at::set_num_interop_threads(static_cast<int>(value));
        break;
      case ATEN_CUDNN_ENABLED:
        context.setUserEnabledCuDNN(on);
        break;
      case ATEN_CUDNN_BENCHMARK:
        context.setBenchmarkCuDNN(on);
        break;
      case ATEN_CUDNN_BENCHMARK_LIMIT:
        context.setBenchmarkLimitCuDNN(static_cast<int>(value));
        break;
      case ATEN_CUDNN_DETERMINISTIC:
        context.setDeterministicCuDNN(on);
        break;
      case ATEN_CUDNN_ALLOW_TF32:
        context.setAllowTF32CuDNN(on);
        break;
      case ATEN_CUBLAS_ALLOW_TF32:
        context.setAllowTF32CuBLAS(on);
        break;
      case ATEN_CUBLAS_ALLOW_FP16_ACCUMULATION:
        context.setAllowFP16AccumulationCuBLAS(on);
        break;
      case ATEN_FLOAT32_MATMUL_PRECISION: {
        static const char* names[] = {"highest", "high", "medium"};
        TORCH_CHECK(value >= 0 && value < 3, "no float32 matmul precision ", value);
        context.setFloat32MatmulPrecision(names[value]);
        break;
      }
      case ATEN_BLAS_BACKEND:
        context.setBlasPreferredBackend(static_cast<at::BlasBackend>(value));
        break;
      case ATEN_LINALG_BACKEND:
        context.setLinalgPreferredBackend(static_cast<at::LinalgBackend>(value));
        break;
      case ATEN_ROCM_FA_BACKEND:
        context.setROCmFAPreferredBackend(static_cast<at::ROCmFABackend>(value));
        break;
      case ATEN_SDP_FLASH:
        context.setSDPUseFlash(on);
        break;
      case ATEN_SDP_FA3:
        context.setSDPUseFA3(on);
        break;
      case ATEN_SDP_FA4:
        context.setSDPUseFA4(on);
        break;
      case ATEN_SDP_MEM_EFFICIENT:
        context.setSDPUseMemEfficient(on);
        break;
      case ATEN_SDP_MATH:
        context.setSDPUseMath(on);
        break;
      case ATEN_SDP_CUDNN:
        context.setSDPUseCuDNN(on);
        break;
      case ATEN_SDP_OVERRIDEABLE:
        context.setSDPUseOverrideable(on);
        break;
      case ATEN_SDP_MATH_FP16_BF16_REDUCTION:
        context.setAllowFP16BF16ReductionMathSDP(on);
        break;
      case ATEN_MKLDNN_ENABLED:
        context.setUserEnabledMkldnn(on);
        break;
      case ATEN_MKLDNN_DETERMINISTIC:
        context.setDeterministicMkldnn(on);
        break;
      case ATEN_NNPACK_ENABLED:
        context.setUserEnabledNNPACK(on);
        break;
      case ATEN_CPU_ALLOW_FP16_REDUCTION:
        context.setAllowFP16ReductionCPU(on);
        break;
      case ATEN_DETERMINISTIC_ALGORITHMS:
        context.setDeterministicAlgorithms(
            on, context.deterministicAlgorithmsWarnOnly());
        break;
      case ATEN_DETERMINISTIC_ALGORITHMS_WARN_ONLY:
        context.setDeterministicAlgorithms(context.deterministicAlgorithms(), on);
        break;
      case ATEN_DETERMINISTIC_FILL_UNINITIALIZED_MEMORY:
        context.setDeterministicFillUninitializedMemory(on);
        break;
      case ATEN_QUANTIZED_ENGINE:
        context.setQEngine(static_cast<at::QEngine>(value));
        break;
      default:
        TORCH_CHECK(false, "no setting ", static_cast<int>(setting));
    }
  });
}

AOTITorchError aten_setting(AtenSetting setting, int64_t* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto& context = at::globalContext();
    switch (setting) {
      case ATEN_NUM_THREADS:
        *ret = at::get_num_threads();
        break;
      case ATEN_NUM_INTEROP_THREADS:
        *ret = static_cast<int64_t>(at::get_num_interop_threads());
        break;
      case ATEN_CUDNN_ENABLED:
        *ret = context.userEnabledCuDNN();
        break;
      case ATEN_CUDNN_BENCHMARK:
        *ret = context.benchmarkCuDNN();
        break;
      case ATEN_CUDNN_BENCHMARK_LIMIT:
        *ret = context.benchmarkLimitCuDNN();
        break;
      case ATEN_CUDNN_DETERMINISTIC:
        *ret = context.deterministicCuDNN();
        break;
      case ATEN_CUDNN_ALLOW_TF32:
        *ret = context.allowTF32CuDNN();
        break;
      case ATEN_CUBLAS_ALLOW_TF32:
        *ret = context.allowTF32CuBLAS();
        break;
      case ATEN_CUBLAS_ALLOW_FP16_ACCUMULATION:
        *ret = context.allowFP16AccumulationCuBLAS();
        break;
      case ATEN_FLOAT32_MATMUL_PRECISION:
        *ret = static_cast<int64_t>(context.float32MatmulPrecision());
        break;
      case ATEN_BLAS_BACKEND:
        *ret = static_cast<int64_t>(context.blasPreferredBackend());
        break;
      case ATEN_LINALG_BACKEND:
        *ret = static_cast<int64_t>(context.linalgPreferredBackend());
        break;
      case ATEN_ROCM_FA_BACKEND:
        *ret = static_cast<int64_t>(context.getROCmFAPreferredBackend());
        break;
      case ATEN_SDP_FLASH:
        *ret = context.userEnabledFlashSDP();
        break;
      case ATEN_SDP_FA3:
        *ret = context.userEnabledFA3SDP();
        break;
      case ATEN_SDP_FA4:
        *ret = context.userEnabledFA4SDP();
        break;
      case ATEN_SDP_MEM_EFFICIENT:
        *ret = context.userEnabledMemEfficientSDP();
        break;
      case ATEN_SDP_MATH:
        *ret = context.userEnabledMathSDP();
        break;
      case ATEN_SDP_CUDNN:
        *ret = context.userEnabledCuDNNSDP();
        break;
      case ATEN_SDP_OVERRIDEABLE:
        *ret = context.userEnabledOverrideableSDP();
        break;
      case ATEN_SDP_MATH_FP16_BF16_REDUCTION:
        *ret = context.allowFP16BF16ReductionMathSDP();
        break;
      case ATEN_MKLDNN_ENABLED:
        *ret = context.userEnabledMkldnn();
        break;
      case ATEN_MKLDNN_DETERMINISTIC:
        *ret = context.deterministicMkldnn();
        break;
      case ATEN_NNPACK_ENABLED:
        *ret = context.userEnabledNNPACK();
        break;
      case ATEN_CPU_ALLOW_FP16_REDUCTION:
        *ret = context.allowFP16ReductionCPU();
        break;
      case ATEN_DETERMINISTIC_ALGORITHMS:
        *ret = context.deterministicAlgorithms();
        break;
      case ATEN_DETERMINISTIC_ALGORITHMS_WARN_ONLY:
        *ret = context.deterministicAlgorithmsWarnOnly();
        break;
      case ATEN_DETERMINISTIC_FILL_UNINITIALIZED_MEMORY:
        *ret = context.deterministicFillUninitializedMemory();
        break;
      case ATEN_QUANTIZED_ENGINE:
        *ret = static_cast<int64_t>(context.qEngine());
        break;
      default:
        TORCH_CHECK(false, "no setting ", static_cast<int>(setting));
    }
  });
}

// --- Build ------------------------------------------------------------------

AOTITorchError aten_has(AtenFeature feature, bool* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    switch (feature) {
      case ATEN_HAS_CUDA:
        *ret = at::hasCUDA();
        break;
      case ATEN_HAS_HIP:
        *ret = at::hasHIP();
        break;
      case ATEN_HAS_MPS:
        *ret = at::hasMPS();
        break;
      case ATEN_HAS_XPU:
        *ret = at::hasXPU();
        break;
      case ATEN_HAS_CUDNN:
        *ret = at::detail::getCUDAHooks().hasCuDNN();
        break;
      case ATEN_HAS_MKL:
        *ret = at::hasMKL();
        break;
      case ATEN_HAS_MKLDNN:
        *ret = at::hasMKLDNN();
        break;
      case ATEN_HAS_OPENMP:
        *ret = at::hasOpenMP();
        break;
      case ATEN_HAS_LAPACK:
        *ret = at::hasLAPACK();
        break;
      case ATEN_HAS_MAGMA:
        *ret = at::hasMAGMA();
        break;
      case ATEN_HAS_KLEIDIAI:
        *ret = at::hasKleidiAI();
        break;
      default:
        TORCH_CHECK(false, "no feature ", static_cast<int>(feature));
    }
  });
}

AOTITorchError aten_library_version(
    AtenLibrary library,
    int64_t* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto& hooks = at::detail::getCUDAHooks();
    switch (library) {
      case ATEN_CUDNN_VERSION:
        *ret = hooks.versionCuDNN();
        break;
      case ATEN_CUDA_RUNTIME_VERSION:
        *ret = hooks.versionCUDART();
        break;
      case ATEN_HIPBLASLT_VERSION:
        *ret = hooks.versionHipBLASLt();
        break;
      default:
        TORCH_CHECK(false, "no library ", static_cast<int>(library));
    }
  });
}

AOTITorchError aten_info(AtenInfo info, StringHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    switch (info) {
      case ATEN_CONFIG:
        *ret = new_string(at::show_config());
        break;
      case ATEN_PARALLEL_INFO:
        *ret = new_string(at::get_parallel_info());
        break;
      case ATEN_CPU_CAPABILITY:
        *ret = new_string(at::get_cpu_capability());
        break;
      case ATEN_MKL_VERSION:
        *ret = new_string(at::get_mkl_version());
        break;
      case ATEN_MKLDNN_VERSION:
        *ret = new_string(at::get_mkldnn_version());
        break;
      case ATEN_OPENMP_VERSION:
        *ret = new_string(at::get_openmp_version());
        break;
      default:
        TORCH_CHECK(false, "no information ", static_cast<int>(info));
    }
  });
}

// --- Interchange ------------------------------------------------------------

AOTITorchError aten_tensor_to_string(
    AtenTensorHandle tensor,
    StringHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::ostringstream out;
    out << *tensor_handle_to_tensor_pointer(tensor);
    *ret = new_string(out.str());
  });
}

AOTITorchError aten_to_dlpack(
    AtenTensorHandle tensor,
    DLManagedTensorVersioned** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = at::toDLPackVersioned(*tensor_handle_to_tensor_pointer(tensor)); });
}

AOTITorchError aten_from_dlpack(
    DLManagedTensorVersioned* capsule,
    AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = new_tensor_handle(at::fromDLPackVersioned(capsule)); });
}

AOTITorchError aten_to_dlpack_unversioned(
    AtenTensorHandle tensor,
    DLManagedTensor** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = at::toDLPack(*tensor_handle_to_tensor_pointer(tensor)); });
}

AOTITorchError aten_from_dlpack_unversioned(
    DLManagedTensor* capsule,
    AtenTensorHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret = new_tensor_handle(at::fromDLPack(capsule)); });
}

AOTITorchError aten_save(
    const char* const* names,
    const AtenTensorHandle* handles,
    int64_t len,
    StringHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::Dict<std::string, at::Tensor> dict;
    for (int64_t i = 0; i < len; ++i) {
      dict.insert(names[i], *tensor_handle_to_tensor_pointer(handles[i]));
    }
    auto bytes = torch::jit::pickle_save(c10::IValue(dict));
    *ret = new_string(std::string(bytes.begin(), bytes.end()));
  });
}

AOTITorchError aten_load(
    const void* data,
    size_t size,
    StringHandle** names,
    AtenTensorHandle** handles,
    int64_t* len) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto* bytes = static_cast<const char*>(data);
    auto value = torch::jit::pickle_load(std::vector<char>(bytes, bytes + size));
    std::vector<std::string> keys;
    std::vector<at::Tensor> values;
    auto add = [&](std::string key, const c10::IValue& item) {
      TORCH_CHECK(item.isTensor(), key, " is a ", item.tagKind(), ", not a tensor");
      keys.push_back(std::move(key));
      values.push_back(item.toTensor());
    };
    if (value.isTensor()) {
      add("", value);
    } else if (value.isGenericDict()) {
      for (const auto& entry : value.toGenericDict()) {
        TORCH_CHECK(entry.key().isString(), "a key is a ", entry.key().tagKind());
        add(entry.key().toStringRef(), entry.value());
      }
    } else if (value.isList() || value.isTuple()) {
      auto items = value.isList() ? value.toListRef().vec()
                                  : value.toTupleRef().elements().vec();
      for (size_t i = 0; i < items.size(); ++i) {
        add(std::to_string(i), items[i]);
      }
    } else {
      TORCH_CHECK(false, "torch.save wrote a ", value.tagKind(), ", not tensors");
    }
    int64_t names_len = 0;
    new_strings(keys, names, &names_len);
    aten_tensor_list(std::move(values), handles, len);
  });
}

// --- AOTInductor packages ---------------------------------------------------

AOTITorchError aten_load_aoti_package(
    const char* path,
    const char* model_name,
    bool run_single_threaded,
    int64_t num_runners,
    int32_t device_index,
    AtenAotiPackageHandle* ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *ret = reinterpret_cast<AtenAotiPackageHandle>(
        new torch::inductor::AOTIModelPackageLoader(
            path,
            model_name,
            run_single_threaded,
            static_cast<size_t>(num_runners),
            static_cast<c10::DeviceIndex>(device_index)));
  });
}

AOTITorchError aten_delete_aoti_package(
    AtenAotiPackageHandle handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({ delete package(handle); });
}

AOTITorchError aten_aoti_package_run(
    AtenAotiPackageHandle handle,
    const AtenTensorHandle* inputs,
    int64_t inputs_len,
    void* stream,
    AtenTensorHandle** ret,
    int64_t* ret_len) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    aten_tensor_list(
        package(handle)->run(tensors(inputs, inputs_len), stream), ret, ret_len);
  });
}

AOTITorchError aten_aoti_package_metadata(
    AtenAotiPackageHandle handle,
    StringHandle** keys,
    StringHandle** values,
    int64_t* len) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::vector<std::string> names, entries;
    for (const auto& [name, entry] : package(handle)->get_metadata()) {
      names.push_back(name);
      entries.push_back(entry);
    }
    new_strings(names, keys, len);
    new_strings(entries, values, len);
  });
}

AOTITorchError aten_aoti_package_call_spec(
    AtenAotiPackageHandle handle,
    StringHandle** ret,
    int64_t* ret_len) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { new_strings(package(handle)->get_call_spec(), ret, ret_len); });
}

AOTITorchError aten_aoti_package_constant_names(
    AtenAotiPackageHandle handle,
    StringHandle** ret,
    int64_t* ret_len) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { new_strings(package(handle)->get_constant_fqns(), ret, ret_len); });
}

AOTITorchError aten_aoti_package_load_constants(
    AtenAotiPackageHandle handle,
    const char* const* names,
    const AtenTensorHandle* handles,
    int64_t len,
    bool use_inactive,
    bool check_full_update,
    bool user_managed) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::unordered_map<std::string, at::Tensor> constants;
    for (int64_t i = 0; i < len; ++i) {
      constants.emplace(names[i], *tensor_handle_to_tensor_pointer(handles[i]));
    }
    package(handle)->load_constants(
        constants, use_inactive, check_full_update, user_managed);
  });
}

// --- Warnings ---------------------------------------------------------------

void aten_set_warning_handler(
    AtenWarningHandler handler,
    void* context) {
  warning_handler.handler = handler;
  warning_handler.context = context;
  // c10 falls back to its stderr handler when the thread's is null.
  c10::WarningUtils::set_warning_handler(handler ? &warning_handler : nullptr);
}
