#ifndef H_ITERA_PONG_H
#define H_ITERA_PONG_H

#include <raylib.h>
#include <iostream>
#include <math/matrix.h>
#include <network/neuralnetwork.h>

#define GAME_WIDTH 600
#define GAME_HEIGHT 600

namespace game
{
    typedef struct Paddle
    {
        Rectangle rect;
        Color color;
        float dy;
    } Paddle;

    typedef struct Ball
    {
        Rectangle rect;
        Color color;
        float dx;
        float dy;
    } Ball;

    class Pong
    {
    public:
        Pong(int game_width, int game_height)
            : game_width(game_width), game_height(game_height)
        {
            default_properties();
        }

        Pong(int game_width, int game_height, network::NeuralNetwork &ai)
            : game_width(game_width), game_height(game_height), ai(ai)
        {
            default_properties();
        }

        double Play(bool render = 1)
        {
            setup(render);
            int num_iters = 0;
            while (running)
            {
                if (++num_iters == max_iters)
                    break;

                if (render && WindowShouldClose())
                    break;

                update_ball();
                update_state(state);
                update_player();

                running = !should_quit();

                if (!render)
                {
                    continue;
                }

                BeginDrawing();
                ClearBackground(bg_color);
                DrawRectangleRec(player.rect, player.color);
                DrawRectangleRec(ball.rect, ball.color);
                EndDrawing();
            }

            if (render)
            {
                CloseWindow();
            }

            return score;
        }

    private:
        int game_width;
        int game_height;
        int fps;
        Color bg_color;
        bool running;
        Paddle player;
        Ball ball;
        network::NeuralNetwork ai;
        math::Matrix ai_output;
        math::Matrix state;
        double score;
        int max_iters;

        void default_properties()
        {
            fps = 30;
            bg_color = {1, 1, 2, 255};
            running = false;
            max_iters = 1000;
        }

        void setup(bool render)
        {
            if (render)
            {
                InitWindow(game_width, game_height, "Pong");
                SetTargetFPS(fps);
            }

            running = true;
            score = 0;

            math::matrix_alloc(ai_output, 1, 1);
            math::matrix_alloc(state, 1, 6);

            setup_player();
            setup_ball();
            setup_ai();
        }

        void setup_player()
        {
            player.rect.width = game_width * 0.025;
            player.rect.height = game_height * 0.15;
            player.rect.x = game_width / 10;
            player.rect.y = std::rand() % game_height - player.rect.height;
            player.color = {103, 179, 181, 255};
            player.dy = 0;
        }

        void setup_ball()
        {
            ball.rect.width = game_width * 0.02;
            ball.rect.height = ball.rect.width;
            ball.rect.x = game_width / 2 - ball.rect.width / 2;
            ball.rect.y = game_height / 2 - ball.rect.height / 2;
            ball.color = {243, 25, 25, 255};

            std::srand(static_cast<unsigned>(std::time(0)));

            ball.dx = (std::rand() % 2 == 0 ? 1 : -1) * (game_width * 0.0375 + std::rand() % game_width * 0.015);
            ball.dy = (std::rand() % 2 == 0 ? 1 : -1) * (game_height * 0.015 + std::rand() % game_height * 0.015);
        }

        void setup_ai()
        {
            if (!ai.Is_Empty())
                return;

            std::cout << "Initializing default AI..." << std::endl;

            ai.Add_Layer(network::DenseLayer(6, 12, "sigmoid"));
            ai.Add_Layer(network::DenseLayer(12, 1, "tanh"));
        }

        void update_player()
        {
            ai.forward(state, ai_output);
            double move = ai_output.data[0];

            player.dy = move * (game_height * 0.0375); // assuming the ai output in interval [-1, +1]

            player.rect.y += player.dy;

            if (player.rect.y < 0)
                player.rect.y = 0;
            if (player.rect.y + player.rect.height > game_height)
                player.rect.y = game_height - player.rect.height;
        }

        void update_ball()
        {
            if (ball.rect.x + ball.rect.width > game_width)
            {
                ball.dx *= -1;
            }

            if (ball.rect.y + ball.rect.height > game_height)
            {
                ball.dy *= -1;
            }

            if (ball.rect.y < 0)
            {
                ball.dy *= -1;
            }

            if (ball.dx < 0 && CheckCollisionRecs(player.rect, ball.rect))
            {
                on_collide();
            }

            ball.rect.x += ball.dx;
            ball.rect.y += ball.dy;
        }

        double ball_speed()
        {
            return sqrt(ball.dx * ball.dx + ball.dy * ball.dy);
        }

        void on_collide()
        {
            float paddle_center = player.rect.y + player.rect.height / 2;
            float ball_center = ball.rect.y + ball.rect.height / 2;

            float impact_position = (ball_center - paddle_center) / (player.rect.height / 2);

            ball.dx *= -1;

            ball.dy = impact_position * (game_height * 0.03) + ball.dy * 0.5f;

            ball.dx *= 1.05f;
            ball.dy *= 1.05f;

            if (std::abs(ball.dy) < 1.0f)
            {
                ball.dy += (ball.dy < 0) ? -5.0f : 5.0f;
            }

            score += ball_speed() * 0.001;
        }

        void update_state(math::Matrix &state)
        {
            assert(state.rows == 1);
            assert(state.cols == 6);

            state.data[0] = player.rect.y;
            state.data[1] = ball.rect.x - player.rect.x;
            state.data[2] = abs(ball.rect.y - player.rect.y);
            state.data[3] = player.dy > 0 ? 0 : 1;
            state.data[4] = ball.dx > 0 ? 0 : 1;
            state.data[5] = ball.dy > 0 ? 0 : 1;

            normalize_state(state);
        }

        void normalize_state(math::Matrix &state)
        {
            assert(state.rows == 1);
            assert(state.cols == 6);

            state.data[0] = math::map_value(state.data[0],
                                            0,
                                            game_height - player.rect.height,
                                            -1,
                                            1);

            state.data[1] = math::map_value(state.data[1],
                                            0,
                                            game_width - ball.rect.width - player.rect.x,
                                            -1,
                                            1);

            state.data[2] = math::map_value(state.data[2],
                                            0,
                                            game_height - ball.rect.height,
                                            -1,
                                            1);
        }

        bool should_quit()
        {
            return ball.rect.x <= 0;
        }
    };
}

#endif // H_ITERA_PONG_H