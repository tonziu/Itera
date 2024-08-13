#ifndef H_ITERA_PONG_H
#define H_ITERA_PONG_H

#include <raylib.h>
#include <iostream>
#include <math/matrix.h>
#include <network/neuralnetwork.h>

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

        void Play(bool render = 1)
        {
            setup(render);
            while (running)
            {
                if (render && WindowShouldClose()) break;

                update_state(state);
                update_player();
                update_ball();

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

        void default_properties()
        {
            fps = 30;
            bg_color = RAYWHITE;
            running = false;
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
            player.rect.width = 15;
            player.rect.height = 60;
            player.rect.x = game_width / 10;
            player.rect.y = game_height / 2 - player.rect.height / 2;
            player.color = BLACK;
            player.dy = 0;
        }

        void setup_ball()
        {
            ball.rect.width = 10;
            ball.rect.height = 10;
            ball.rect.x = game_width / 2 - ball.rect.width/2;
            ball.rect.y = game_height / 2 - ball.rect.height / 2;
            ball.color = BLACK;
            ball.dx = 7;
            ball.dy = 8;
        }

        void setup_ai()
        {
            ai.Add_Layer(network::DenseLayer(6, 12, math::matrix_sigmoid_in_place));
            ai.Add_Layer(network::DenseLayer(12, 1, math::matrix_tanh_in_place));
        }

        void update_player()
        {
            ai.forward(state, ai_output);
            double move = ai_output.data[0];

            // Let's assume the AI output is in range [-1, 1], where:
            // -1 means move up, 1 means move down, 0 means stay
            player.dy = move * 7.0f;  // Scale the movement speed

            player.rect.y += player.dy;

            // Keep the paddle within the screen boundaries
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

            if (CheckCollisionRecs(player.rect, ball.rect))
            {
                on_collide();
            }

            ball.rect.x += ball.dx;
            ball.rect.y += ball.dy;
        }

        void on_collide()
        {
            ball.dx *= -1;
            score += 1.0;
        }

        void update_state(math::Matrix& state)
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

        void normalize_state(math::Matrix& state)
        {
            assert(state.rows == 1);
            assert(state.cols == 6);

            // Player pos y
            state.data[0] = math::map_value(state.data[0],
                                            0, 
                                            game_height - player.rect.height,
                                            -1,
                                            1);
            
            // Horizontal distance player-ball
            state.data[1] = math::map_value(state.data[1],
                                            0, 
                                            game_width - ball.rect.width - player.rect.x,
                                            -1,
                                            1);

            // Vertical distance player-ball
            state.data[2] = math::map_value(state.data[2],
                                            0,
                                            game_height-ball.rect.height,
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