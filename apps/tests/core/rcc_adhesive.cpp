#include <catch2/catch_all.hpp>
#include <uipc/uipc.h>
#include <uipc/constitution/rcc_adhesive.h>
#include <uipc/builtin/constitution_uid_collection.h>

TEST_CASE("rcc_adhesive", "[contact_model][rcc_adhesive]")
{
    using namespace uipc;
    using namespace uipc::core;
    using namespace uipc::constitution;

    SECTION("apply_to registers all 8 adhesion attributes")
    {
        Scene scene;
        auto& tabular = scene.contact_tabular();

        RCCAdhesive adhesive;
        adhesive.apply_to(tabular);

        auto models = tabular.contact_models();
        REQUIRE(models.find<Float>("Cn"));
        REQUIRE(models.find<Float>("Ct"));
        REQUIRE(models.find<Float>("W"));
        REQUIRE(models.find<Float>("eta"));
        REQUIRE(models.find<Float>("bonding_rate"));
        REQUIRE(models.find<Float>("p0"));
        REQUIRE(models.find<Float>("initial_beta"));
        REQUIRE(models.find<IndexT>("adhesion_enabled"));
    }

    SECTION("apply_to is idempotent")
    {
        Scene scene;
        auto& tabular = scene.contact_tabular();

        RCCAdhesive adhesive;
        adhesive.apply_to(tabular);
        adhesive.apply_to(tabular);  // must not throw / clobber

        auto models = tabular.contact_models();
        REQUIRE(models.find<Float>("Cn"));
    }

    SECTION("default_model writes row 0")
    {
        Scene scene;
        auto& tabular = scene.contact_tabular();

        RCCAdhesive adhesive;
        adhesive.default_model(tabular,
                               /*Cn=*/1.5e6,
                               /*Ct=*/2.5e6,
                               /*W=*/0.75,
                               /*eta=*/3.0,
                               /*bonding_rate=*/0.5,
                               /*p0=*/0.25,
                               /*initial_beta=*/0.1,
                               /*enabled=*/true);

        auto models = tabular.contact_models();
        REQUIRE(models.find<Float>("Cn")->view()[0] == Catch::Approx(1.5e6));
        REQUIRE(models.find<Float>("Ct")->view()[0] == Catch::Approx(2.5e6));
        REQUIRE(models.find<Float>("W")->view()[0] == Catch::Approx(0.75));
        REQUIRE(models.find<Float>("eta")->view()[0] == Catch::Approx(3.0));
        REQUIRE(models.find<Float>("bonding_rate")->view()[0] == Catch::Approx(0.5));
        REQUIRE(models.find<Float>("p0")->view()[0] == Catch::Approx(0.25));
        REQUIRE(models.find<Float>("initial_beta")->view()[0] == Catch::Approx(0.1));
        REQUIRE(models.find<IndexT>("adhesion_enabled")->view()[0] == 1);
    }

    SECTION("set writes a specific (L, R) pair")
    {
        Scene scene;
        auto& tabular = scene.contact_tabular();

        auto wood   = tabular.create("wood");
        auto rubber = tabular.create("rubber");
        tabular.insert(wood, rubber, 0.3, 1e8);

        RCCAdhesive adhesive;
        adhesive.set(tabular, wood, rubber,
                     /*Cn=*/9e5,
                     /*Ct=*/8e5,
                     /*W=*/0.6,
                     /*eta=*/2.5,
                     /*bonding_rate=*/0.4,
                     /*p0=*/0.15,
                     /*initial_beta=*/0.2,
                     /*enabled=*/true);

        auto models = tabular.contact_models();
        auto topo   = models.find<Vector2i>("topo")->view();
        auto cn     = models.find<Float>("Cn")->view();
        auto ct     = models.find<Float>("Ct")->view();
        auto en     = models.find<IndexT>("adhesion_enabled")->view();

        // locate the row for (wood, rubber)
        Vector2i want{wood.id(), rubber.id()};
        if(want.x() > want.y())
            std::swap(want.x(), want.y());

        IndexT idx = -1;
        for(IndexT i = 0; i < static_cast<IndexT>(topo.size()); ++i)
            if(topo[i].x() == want.x() && topo[i].y() == want.y())
                idx = i;
        REQUIRE(idx > 0);  // not the default row

        REQUIRE(cn[idx] == Catch::Approx(9e5));
        REQUIRE(ct[idx] == Catch::Approx(8e5));
        REQUIRE(en[idx] == 1);

        // unspecified rows should retain their default 0
        REQUIRE(cn[0] == Catch::Approx(0.0));
    }

    SECTION("disabled flag stored as 0")
    {
        Scene scene;
        auto& tabular = scene.contact_tabular();

        RCCAdhesive adhesive;
        adhesive.default_model(tabular, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, /*enabled=*/false);

        REQUIRE(tabular.contact_models().find<IndexT>("adhesion_enabled")->view()[0] == 0);
    }

    SECTION("UID is registered in ConstitutionUIDCollection")
    {
        RCCAdhesive adhesive;
        auto&       reg = builtin::ConstitutionUIDCollection::instance();
        REQUIRE(reg.exists(adhesive.get_uid()));
        const auto& info = reg.find(adhesive.get_uid());
        REQUIRE(info.name == "RCCAdhesive");
        REQUIRE(info.type == "ContactModel");
    }
}
