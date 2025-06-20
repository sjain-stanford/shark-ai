#include <sharkfuser.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("sharkfuser::add adds two numbers", "[add]") {
  REQUIRE(sharkfuser::add(2, 3) == 5);
  REQUIRE(sharkfuser::add(-1, 1) == 0);
}
