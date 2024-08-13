#include <iostream>

#include <math/matrix.h>
#include <network/denselayer.h>
#include <network/neuralnetwork.h>
#include <game/pong.h>

using namespace math;

int main(void)
{
    game::Pong pong(600, 600);
    pong.Play(true);

    return 0;
}