#include <catch2/catch_all.hpp>

int main(int argc, char* argv[])
{
    Catch::Session session;
    const int      result = session.applyCommandLine(argc, argv);
    if(result != 0)
        return result;
    return session.run();
}
