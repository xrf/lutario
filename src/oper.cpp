#include "alloc.hpp"
#include "basis.hpp"
#include "oper.hpp"

PtrAllocReq<double> Oper::alloc_req(const ManyBodyBasis &basis, OperKind kk)
{
    *this = Oper(nullptr, basis, kk);
    return {&this->_data, this->size()};
}

PtrAllocReq<double> ManyBodyOper::alloc_req(const ManyBodyBasis &basis)
{
    *this = ManyBodyOper(nullptr, basis);
    return {&this->_data, this->size()};
}
