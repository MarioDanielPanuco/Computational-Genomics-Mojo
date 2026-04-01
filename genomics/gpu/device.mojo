"""
GPU device context wrapper for the genomics library.

Provides a thin, typed layer over MAX's DeviceContext for:
  - Typed buffer allocation and host <-> device transfers
  - Batch sequence upload (SequenceBatch → device buffers)
  - Availability guard (comptime has_accelerator check)
"""
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from genomics.core.sequence import SequenceBatch
from genomics.core.dna import NMask


def check_gpu_available():
    """Compile-time assertion that a GPU accelerator is present."""
    comptime assert has_accelerator(), "No GPU accelerator found — GPU kernels require a GPU."


struct GenomicsDevice(Movable):
    """Thin wrapper around DeviceContext providing typed allocation helpers."""

    var ctx: DeviceContext

    def __init__(out self) raises:
        comptime if not has_accelerator():
            comptime assert False, "GenomicsDevice requires a GPU accelerator."
        self.ctx = DeviceContext()

    def synchronize(mut self) raises:
        self.ctx.synchronize()

    # ===------------------------------------------------------------------=== #
    # Typed allocation
    # ===------------------------------------------------------------------=== #

    def alloc_uint64(mut self, count: Int) raises -> DeviceBuffer[DType.uint64]:
        return self.ctx.enqueue_create_buffer[DType.uint64](count)

    def alloc_uint8(mut self, count: Int) raises -> DeviceBuffer[DType.uint8]:
        return self.ctx.enqueue_create_buffer[DType.uint8](count)

    def alloc_float32(mut self, count: Int) raises -> DeviceBuffer[DType.float32]:
        return self.ctx.enqueue_create_buffer[DType.float32](count)

    def alloc_int32(mut self, count: Int) raises -> DeviceBuffer[DType.int32]:
        return self.ctx.enqueue_create_buffer[DType.int32](count)

    def alloc_host_uint64(mut self, count: Int) raises -> DeviceBuffer[DType.uint64]:
        return self.ctx.enqueue_create_host_buffer[DType.uint64](count)

    def alloc_host_float32(mut self, count: Int) raises -> DeviceBuffer[DType.float32]:
        return self.ctx.enqueue_create_host_buffer[DType.float32](count)

    # ===------------------------------------------------------------------=== #
    # SequenceBatch upload
    # ===------------------------------------------------------------------=== #

    def upload_batch_packed(
        mut self,
        batch: SequenceBatch,
    ) raises -> DeviceBuffer[DType.uint64]:
        """Upload the packed UInt64 words of a SequenceBatch to the GPU.

        Returns a device buffer containing all packed words concatenated.
        The caller is responsible for also uploading offsets and lengths
        so kernels can find each sequence.
        """
        var n_words = len(batch.packed)
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.uint64](n_words)

        # Fill host buffer from batch
        with host_buf.map_to_host() as mapped:
            for i in range(n_words):
                mapped[i] = batch.packed[i]

        var dev_buf = self.ctx.enqueue_create_buffer[DType.uint64](n_words)
        self.ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return dev_buf

    def upload_lengths(
        mut self,
        batch: SequenceBatch,
    ) raises -> DeviceBuffer[DType.int32]:
        """Upload sequence lengths as Int32 to the GPU."""
        var n = batch.count
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.int32](n)
        with host_buf.map_to_host() as mapped:
            for i in range(n):
                mapped[i] = batch.lengths[i]
        var dev_buf = self.ctx.enqueue_create_buffer[DType.int32](n)
        self.ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return dev_buf

    def upload_offsets(
        mut self,
        batch: SequenceBatch,
    ) raises -> DeviceBuffer[DType.int32]:
        """Upload word offsets as Int32 to the GPU."""
        var n = batch.count
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.int32](n)
        with host_buf.map_to_host() as mapped:
            for i in range(n):
                mapped[i] = batch.offsets[i]
        var dev_buf = self.ctx.enqueue_create_buffer[DType.int32](n)
        self.ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return dev_buf

    def download_float32(
        mut self,
        dev_buf: DeviceBuffer[DType.float32],
        count: Int,
    ) raises -> List[Float32]:
        """Download a Float32 device buffer to a host List."""
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.float32](count)
        self.ctx.enqueue_copy(dst_buf=host_buf, src_buf=dev_buf)
        self.ctx.synchronize()
        var result = List[Float32](capacity=count)
        with host_buf.map_to_host() as mapped:
            for i in range(count):
                result.append(Float32(mapped[i]))
        return result

    def download_uint64(
        mut self,
        dev_buf: DeviceBuffer[DType.uint64],
        count: Int,
    ) raises -> List[UInt64]:
        """Download a UInt64 device buffer to a host List."""
        var host_buf = self.ctx.enqueue_create_host_buffer[DType.uint64](count)
        self.ctx.enqueue_copy(dst_buf=host_buf, src_buf=dev_buf)
        self.ctx.synchronize()
        var result = List[UInt64](capacity=count)
        with host_buf.map_to_host() as mapped:
            for i in range(count):
                result.append(UInt64(mapped[i]))
        return result
