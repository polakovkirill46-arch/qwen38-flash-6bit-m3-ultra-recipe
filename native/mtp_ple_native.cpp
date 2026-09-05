// Experimental deferred PLE gather. Local integration prototype, 2026.
// CPU primitive scheduling pattern adapted from Apple's MLX axpby example
// (Copyright 2023-2025 Apple Inc., MIT). Deferred PLE motivation: David Dalcu /
// mlx-serve PR350 (MIT); no mutable unfilled-leaf implementation copied.
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/common/utils.h"
namespace mx=mlx::core;
namespace nb=nanobind;
struct Mapping {
 void* ptr=MAP_FAILED; size_t bytes=0;
 explicit Mapping(const std::string& path) {
  int fd=open(path.c_str(),O_RDONLY);if(fd<0)throw std::runtime_error("PLE open failed");
  struct stat st{};
  if(fstat(fd,&st)!=0 || !S_ISREG(st.st_mode) || st.st_size<=0){close(fd);throw std::runtime_error("PLE invalid file");}
  bytes=size_t(st.st_size);ptr=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
  if(ptr==MAP_FAILED)throw std::runtime_error("PLE mmap failed");
 }
 ~Mapping(){if(ptr!=MAP_FAILED)munmap(ptr,bytes);}
 Mapping(const Mapping&)=delete;
};
struct Bank {
 std::vector<std::shared_ptr<Mapping>> maps;
 std::vector<uint64_t> byte_offsets,ends;
 size_t dims;
 Bank(const std::vector<std::string>& paths,const std::vector<uint64_t>& offsets,
      const std::vector<uint64_t>& counts,size_t dim):byte_offsets(offsets),dims(dim){
  if(paths.empty() || paths.size()!=offsets.size() || paths.size()!=counts.size() || dim==0 || dim>10240)
   throw std::invalid_argument("PLE bad bank metadata");
  uint64_t total=0;
  for(size_t i=0;i<paths.size();++i){
   auto m=std::make_shared<Mapping>(paths[i]);
   if(counts[i]==0 || offsets[i]>m->bytes || counts[i]>(m->bytes-offsets[i])/(dim*2) || counts[i]>UINT64_MAX-total)
    throw std::invalid_argument("PLE shard outside file bounds");
   maps.push_back(m);total+=counts[i];ends.push_back(total);
  }
 }
};
class Gather final:public mx::Primitive {
 std::shared_ptr<Bank> bank_;
 public:
 Gather(mx::Stream s,std::shared_ptr<Bank> b):mx::Primitive(s),bank_(std::move(b)){}
 const char* name()const override{return "FlashDeferredPLE";}
 void eval_gpu(const std::vector<mx::array>&,std::vector<mx::array>&)override{
  throw std::runtime_error("Deferred PLE must execute on CPU stream");
 }
 void eval_cpu(const std::vector<mx::array>& ins,std::vector<mx::array>& outs)override{
  const auto& x=ins[0];auto& out=outs[0];
  out.set_data(mx::allocator::malloc(out.nbytes()));
  auto& enc=mx::cpu::get_command_encoder(stream());enc.set_input_array(x);enc.set_output_array(out);
  // Own arrays AND the mmap bank until the queued closure completes. No Python
  // object/GIL is accessed by the worker. IDs are not read at graph-build time.
  enc.dispatch([x,out,bank=bank_]()mutable{
   const auto* ids=x.data<int64_t>();auto* dest=out.data<mx::bfloat16_t>();
   for(size_t i=0;i<x.size();++i){
    int64_t id=ids[mx::elem_to_loc(i,x.shape(),x.strides())];
    if(id<0 || uint64_t(id)>=bank->ends.back())throw std::out_of_range("PLE index out of range");
    auto it=std::upper_bound(bank->ends.begin(),bank->ends.end(),uint64_t(id));
    size_t shard=size_t(it-bank->ends.begin());uint64_t start=shard?bank->ends[shard-1]:0;
    const auto* src=static_cast<const char*>(bank->maps[shard]->ptr)+bank->byte_offsets[shard]+(uint64_t(id)-start)*bank->dims*2;
    std::memcpy(dest+i*bank->dims,src,bank->dims*2);
   }
  });
 }
};
mx::array gather(std::shared_ptr<Bank> bank,const mx::array& indices){
 if(indices.size()==0 || indices.size()>256)throw std::invalid_argument("PLE prototype accepts 1..256 indices");
 auto stream=mx::default_stream(mx::Device(mx::Device::cpu));
 auto ids=mx::astype(indices,mx::int64,stream);
 auto shape=indices.shape();shape.push_back(int(bank->dims));
 return mx::array(shape,mx::bfloat16,std::make_shared<Gather>(stream,std::move(bank)),{ids});
}
NB_MODULE(_mtp_ple_native,m){
 m.attr("BUILT_AGAINST_MLX")="0.32.2";
 nb::class_<Bank>(m,"Bank").def(nb::init<const std::vector<std::string>&,const std::vector<uint64_t>&,const std::vector<uint64_t>&,size_t>());
 m.def("gather",&gather);
}
