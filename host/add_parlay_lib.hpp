#include "parlay/primitives.h"
#include <climits>
using namespace std;

namespace parlay {
// Return a sequence consisting of the indices such that
// b[i] is true, or is implicitly convertible to true.

template <PARLAY_RANGE_TYPE BoolSeq, PARLAY_RANGE_TYPE R_out>
auto pack_index_into(const BoolSeq& b, R_out&& out) {
    using Idx_Type = uint32_t;
    static_assert(std::is_convertible<decltype(*std::begin(b)), bool>::value);
    
    assert(b.size() < UINT32_MAX);
    auto identity = [](size_t i) -> Idx_Type { return static_cast<Idx_Type>(i); };
    // return pack(delayed_seq<Idx_Type>(b.size(), identity), b, no_flag);
    return internal::pack_out(make_slice(delayed_seq<Idx_Type>(b.size(), identity)), b, make_slice(out));
}

}  // namespace parlay