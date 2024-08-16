#ifndef H_ITERA_PONG_H
#define H_ITERA_PONG_H

#include <raylib.h>
#include <iostream>
#include <math/matrix.h>
#include <network/neuralnetwork.h>
#include <common.h>

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

        double Play(common::SimInfo& info)
        {
            setup();
            int num_iters = 0;
            while (running)
            {
                if (++num_iters == max_iters)
                    break;

                if (WindowShouldClose())
                    break;

                if (IsKeyPressed(KEY_S))
                {
                    slowdown = !slowdown;
                    if (slowdown) SetTargetFPS(30);
                    else SetTargetFPS(0);
                }


                update_state(state);
                update_player();
                update_ball();

                running = !should_quit();

                BeginDrawing();
                ClearBackground(bg_color);
                DrawRectangleRec(ball.rect, ball.color);
                std::string gen_text = "Generation: " + std::to_string(info.num_generation);
                std::string score_text = "Score: " + std::to_string(score);
                DrawText(gen_text.c_str(), GAME_WIDTH/2 - MeasureText(gen_text.c_str(), 18) / 2, 50, 18, RAYWHITE);
                DrawText(score_text.c_str(), GAME_WIDTH/2 - MeasureText(score_text.c_str(), 18) / 2, 80, 18, RAYWHITE);
                DrawRectangleRec(player.rect, player.color);
                EndDrawing();
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
        double max_ball_speed;
        double max_paddle_speed;
        bool slowdown;

        void default_properties()
        {
            fps = 30;
            bg_color = {1, 1, 2, 255};
            running = false;
            max_iters = 1000;
            max_ball_speed = game_width / 10;
            max_paddle_speed = game_width / 10;
            slowdown = false;
        }

        void setup()
        {
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

            ball.dx = game_width * 0.0375 + std::rand() % game_width * 0.015;
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

            player.dy = move * (game_height * 0.04); // assuming the ai output in interval [-1, +1]

            if (player.dy > max_paddle_speed) player.dy = max_paddle_speed;
            else if (player.dy < -max_paddle_speed) player.dy = -max_paddle_speed;

            player.rect.y += player.dy;

            if (player.rect.y < 0)
            {
                score -= 0.01;
                player.rect.y = 0;
            }
                
            else if (player.rect.y + player.rect.height > game_height)
            {
                score -= 0.01;
                player.rect.y = game_height - player.rect.height;
            }

            else score += 0.01;
                

            double dist = ball_player_distance();
            if (dist < player.rect.height)
                score += 0.2;
            else
                score -= 0.1;
            if (score < 0)
                score = 0;
        }

        void update_ball()
        {
            if (ball.rect.x + ball.rect.width >= game_width)
            {
                ball.dx *= -1;
            }

            if (ball.rect.y + ball.rect.height >= game_height)
            {
                ball.dy *= -1;
            }

            if (ball.rect.y <= 0)
            {
                ball.dy *= -1;
            }

            if (ball.dx < 0 && CheckCollisionRecs(player.rect, ball.rect))
            {
                on_collide();
            }

            if (ball.dx > max_ball_speed) ball.dx = max_ball_speed;
            else if (ball.dx < -max_ball_speed) ball.dx = -max_ball_speed;

            if (ball.dy > max_ball_speed) ball.dy = max_ball_speed;
            else if (ball.dy < -max_ball_speed) ball.dy = -max_ball_speed;

            ball.rect.x += ball.dx;
            ball.rect.y += ball.dy;
        }

        double ball_speed()
        {
            return sqrt(ball.dx * ball.dx + ball.dy * ball.dy);
        }

        int ball_player_distance()
        {
            return abs((ball.rect.y + ball.rect.height/2) - (player.rect.y + player.rect.height/2));
        }

        void on_collide()
        {
            float paddle_center = player.rect.y + player.rect.height / 2;
            float ball_center = ball.rect.y + ball.rect.height / 2;

            float impact_position = (ball_center - paddle_center) / (player.rect.height / 2);

            ball.dx *= -1;

            ball.dy += impact_position * (std::rand() % 2 == 0 ? -1 : 1);

            ball.dx *= 1.05f;
            ball.dy *= 1.05f;

            if (std::abs(ball.dy) < 1.0f)
            {
                ball.dy += (ball.dy < 0) ? -5.0f : 5.0f;
            }

            score += 1;
        }

        void update_state(math::Matrix &state)
        {
            assert(state.rows == 1);
            assert(state.cols == 6);

            state.data[0] = player.rect.y;
            state.data[1] = player.dy;
            state.data[2] = ball.rect.x;
            state.data[3] = ball.rect.y;
            state.data[4] = ball.dx;
            state.data[5] = ball.dy;
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
                                           -max_paddle_speed,
                                            max_paddle_speed,
                                            -1,
                                            1);

            state.data[2] = math::map_value(state.data[2],
                                            player.rect.x,
                                            game_width - ball.rect.width,
                                            -1,
                                            1);
            
            state.data[3] = math::map_value(state.data[3],
                                            0,
                                            game_height - ball.rect.height,
                                            -1,
                                            1);

            state.data[4] = math::map_value(state.data[4],
                                            -max_ball_speed,
                                            max_ball_speed,
                                            -1,
                                            1);

            state.data[5] = math::map_value(state.data[5],
                                            -max_ball_speed,
                                            max_ball_speed,
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