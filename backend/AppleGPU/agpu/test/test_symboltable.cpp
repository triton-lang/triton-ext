#include "agpu/bind/SymbolTable.h"
#include "harness.h"

using namespace agpu;

int main() {
  CASE("unbound and dataless are different answers");
  {
    SymbolTable t;
    t.bindDataless(1);

    CHECK(t.isBound(1));
    CHECK(t.isDataless(1));
    CHECK(t.namesOf(1) != nullptr);
    CHECK(t.namesOf(1)->empty());

    CHECK(!t.isBound(2));
    CHECK(!t.isDataless(2));
    CHECK(t.namesOf(2) == nullptr);
  }

  CASE("a read does not create an entry");
  {
    SymbolTable t;
    t.bindScalar(1, "v0");
    CHECK_EQ(t.size(), (std::size_t)1);

    CHECK(t.namesOf(99) == nullptr);
    CHECK_EQ(t.regCount(99), (std::size_t)0);
    CHECK(t.regAt(99, 0) == nullptr);
    CHECK(t.scalarName(99) == nullptr);
    CHECK(!t.isDataless(99));
    CHECK_EQ(t.size(), (std::size_t)1);
    CHECK(!t.isBound(99));
  }

  CASE("aliasing copies before inserting");
  {
    SymbolTable t;
    t.bindRegs(1, {"a0", "a1", "a2", "a3"});
    for (ValueId v = 100; v < 200; ++v)
      t.bindScalar(v, "filler" + std::to_string(v));

    CHECK(t.alias(2, 1));
    CHECK(t.namesOf(2) != nullptr);
    CHECK_EQ(*t.namesOf(2), (ValueNames{"a0", "a1", "a2", "a3"}));
    CHECK_EQ(*t.namesOf(1), (ValueNames{"a0", "a1", "a2", "a3"}));
  }

  CASE("aliasing an unbound value fails, leaving it unbound");
  {
    SymbolTable t;
    CHECK(!t.alias(2, 1));
    CHECK(!t.isBound(2));
  }

  CASE("aliasing a dataless value succeeds and stays dataless");
  {
    SymbolTable t;
    t.bindDataless(1);
    CHECK(t.alias(2, 1));
    CHECK(t.isBound(2));
    CHECK(t.isDataless(2));
  }

  CASE("a splat broadcasts: one name answers every register");
  {
    SymbolTable t;
    t.bindScalar(1, "s");
    for (std::size_t r = 0; r < 8; ++r) {
      CHECK(t.regAt(1, r) != nullptr);
      CHECK_EQ(*t.regAt(1, r), msl::Str("s"));
    }
  }

  CASE("a genuine tensor does not broadcast past its end");
  {
    SymbolTable t;
    t.bindRegs(1, {"v0", "v1"});
    CHECK_EQ(*t.regAt(1, 0), msl::Str("v0"));
    CHECK_EQ(*t.regAt(1, 1), msl::Str("v1"));
    CHECK(t.regAt(1, 2) == nullptr);
  }

  CASE("a dataless or unbound value names no register");
  {
    SymbolTable t;
    t.bindDataless(1);
    CHECK(t.regAt(1, 0) == nullptr);
    CHECK(t.regAt(2, 0) == nullptr);
    CHECK(t.scalarName(1) == nullptr);
    CHECK(t.scalarName(2) == nullptr);
  }

  CASE("scalarName answers only for an actual scalar");
  {
    SymbolTable t;
    t.bindScalar(1, "s");
    t.bindRegs(2, {"v0", "v1"});
    CHECK(t.scalarName(1) != nullptr);
    CHECK_EQ(*t.scalarName(1), msl::Str("s"));
    CHECK(t.scalarName(2) == nullptr);
  }

  CASE("regCount is zero for both dataless and unbound");
  {
    SymbolTable t;
    t.bindDataless(1);
    t.bindRegs(2, {"v0", "v1", "v2"});
    CHECK_EQ(t.regCount(1), (std::size_t)0);
    CHECK_EQ(t.regCount(3), (std::size_t)0);
    CHECK_EQ(t.regCount(2), (std::size_t)3);
    CHECK(t.isBound(1));
    CHECK(!t.isBound(3));
  }

  CASE("rebinding replaces the registers");
  {
    SymbolTable t;
    t.bindRegs(1, {"a0", "a1"});
    t.bindRegs(1, {"b0"});
    CHECK_EQ(t.regCount(1), (std::size_t)1);
    CHECK_EQ(*t.regAt(1, 0), msl::Str("b0"));
  }

  CASE("uniformNameOf answers only when every register agrees");
  {
    SymbolTable t;
    t.bindScalar(1, "base");
    t.bindRegs(2, {"base", "base", "base"});
    t.bindRegs(3, {"base", "other"});
    t.bindDataless(4);

    CHECK(t.uniformNameOf(1) && *t.uniformNameOf(1) == msl::Str("base"));
    CHECK(t.uniformNameOf(2) && *t.uniformNameOf(2) == msl::Str("base"));
    CHECK(t.uniformNameOf(3) == nullptr);
    CHECK(t.uniformNameOf(4) == nullptr);
    CHECK(t.uniformNameOf(99) == nullptr);
  }

  CASE("a borrowed name is not the borrower's to declare");
  {
    SymbolTable t;
    t.bindScalar(1, "arg0");
    t.bindScalar(2, "arg0");
    t.bindRegs(3, {"arg0", "arg0"});

    CHECK_EQ(t.ownedNamesOf(1).size(), (std::size_t)1);
    CHECK_EQ(t.ownedNamesOf(1)[0], msl::Str("arg0"));
    CHECK(t.ownsAnyName(1));

    CHECK(t.ownedNamesOf(2).empty());
    CHECK(!t.ownsAnyName(2));
    CHECK(t.ownedNamesOf(3).empty());
    CHECK(!t.ownsAnyName(3));

    CHECK_EQ(*t.regAt(2, 0), msl::Str("arg0"));
    CHECK_EQ(*t.regAt(3, 1), msl::Str("arg0"));
  }

  CASE("ownership survives a rebind and lists each name once");
  {
    SymbolTable t;
    t.bindRegs(1, {"v0", "v1"});
    t.bindRegs(1, {"v0", "v1"});
    CHECK_EQ(t.ownedNamesOf(1).size(), (std::size_t)2);

    t.bindRegs(2, {"w0", "w0", "w0"});
    CHECK_EQ(t.ownedNamesOf(2).size(), (std::size_t)1);

    CHECK(t.alias(3, 1));
    CHECK(t.ownedNamesOf(3).empty());

    t.clear();
    CHECK(t.ownedNamesOf(1).empty());
    t.bindScalar(9, "v0");
    CHECK(t.ownsAnyName(9));
  }

  return ::agpu_test::report("SymbolTable");
}
