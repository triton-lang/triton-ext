// Memdesc handles: index and subslice as TileView operations.
#include "agpu/core/MemDesc.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("a fresh handle is packed row-major");
  {
    MemDesc d = allocMemDesc("buf", {4, 8});
    CHECK_EQ(d.offsetOf({0, 0}), 0);
    CHECK_EQ(d.offsetOf({0, 1}), 1);
    CHECK_EQ(d.offsetOf({1, 0}), 8);
    CHECK_EQ(d.offsetOf({3, 7}), 31);
    CHECK_EQ(d.cosizeElems(), 32);
  }

  CASE("the buffer name travels with the handle");
  {
    MemDesc d = allocMemDesc("pool", {4, 8});
    CHECK_EQ(d.index(1).buffer, std::string("pool"));
    CHECK_EQ(d.subslice({1, 1}, {2, 2}).buffer, std::string("pool"));
  }

  CASE("indexing a multi-buffered allocation picks one slice");
  {
    MemDesc d = allocMultiBuffered("buf", 3, {4, 8});
    CHECK_EQ(d.cosizeElems(), 3 * 32);

    CHECK_EQ(d.index(0).offsetOf({0, 0}), 0);
    CHECK_EQ(d.index(1).offsetOf({0, 0}), 32);
    CHECK_EQ(d.index(2).offsetOf({0, 0}), 64);
  }

  CASE("an indexed slice has the rank its type says");
  {
    // memdesc_index drops the buffer dimension.
    MemDesc slice = allocMultiBuffered("buf", 3, {4, 8}).index(1);
    CHECK_EQ(slice.offsetOf({1, 1}), 32 + 8 + 1);
    CHECK_EQ(slice.view.rank(), 2);
  }

  CASE("indexing twice composes, as the memdescMap rebind does");
  {
    MemDesc d = allocMultiBuffered("buf", 2, {3, 4});
    MemDesc row = d.index(1);    // second half: offset 12
    MemDesc cell = row.index(2); // third row of that: +8
    CHECK_EQ(cell.offsetOf({0}), 12 + 8);
  }

  CASE("a subslice's origin carries the offset");
  {
    MemDesc d = allocMemDesc("buf", {8, 16});
    MemDesc s = d.subslice({2, 4}, {4, 8});
    CHECK_EQ(s.offsetOf({0, 0}), 2 * 16 + 4);
    CHECK_EQ(s.offsetOf({1, 0}), 3 * 16 + 4);
  }

  CASE("a subslice inherits the parent's strides");
  {
    // A 4x8 window into a 16-wide tile still steps 16 per row.
    MemDesc s = allocMemDesc("buf", {8, 16}).subslice({0, 0}, {4, 8});
    CHECK_EQ(s.offsetOf({1, 0}), 16);
    CHECK_EQ(s.offsetOf({0, 1}), 1);
  }

  CASE("a subslice with no extent runs to the end of each dimension");
  {
    MemDesc s = allocMemDesc("buf", {8, 16}).subslice({2, 4});
    CHECK_EQ(s.view.extentAt(0), 6);
    CHECK_EQ(s.view.extentAt(1), 12);
    CHECK_EQ(s.offsetOf({0, 0}), 2 * 16 + 4);
  }

  CASE("subslicing composes, accumulating both offsets");
  {
    MemDesc d = allocMemDesc("buf", {16, 16});
    MemDesc a = d.subslice({4, 4}, {8, 8});
    MemDesc b = a.subslice({2, 2}, {4, 4});
    CHECK_EQ(b.offsetOf({0, 0}), (4 + 2) * 16 + (4 + 2));
  }

  CASE("a zero subslice is the parent's own origin");
  {
    MemDesc d = allocMemDesc("buf", {8, 8}).subslice({1, 1}, {4, 4});
    MemDesc same = d.subslice({0, 0}, {4, 4});
    CHECK_EQ(same.offsetOf({0, 0}), d.offsetOf({0, 0}));
    CHECK_EQ(same.offsetOf({2, 2}), d.offsetOf({2, 2}));
  }

  CASE("indexing then subslicing addresses the right buffer's window");
  {
    MemDesc d = allocMultiBuffered("buf", 2, {8, 16});
    MemDesc window = d.index(1).subslice({2, 4}, {4, 8});
    CHECK_EQ(window.offsetOf({0, 0}), 128 + 2 * 16 + 4);
    CHECK_EQ(window.offsetOf({1, 0}), 128 + 3 * 16 + 4);
  }

  CASE("sizing and addressing come from one object");
  {
    MemDesc d = allocMultiBuffered("buf", 3, {4, 8});
    CHECK_EQ(d.cosizeElems(), 96);
    MemDesc last = d.index(2);
    CHECK(last.offsetOf({3, 7}) < d.cosizeElems());
    CHECK_EQ(last.offsetOf({3, 7}), 95);
  }

  return ::agpu_test::report("MemDesc");
}
