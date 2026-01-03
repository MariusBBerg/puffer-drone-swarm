// Raylib renderer for DroneSwarm (no allocations after init)
#pragma once

#include "drone_swarm.h"
#include "raylib.h"
#include <stdio.h>

typedef struct DroneSwarmClient {
    int width;
    int height;
    int scale;
    bool initialized;
} DroneSwarmClient;

static DroneSwarmClient g_client = {0};

static bool init_render_client(float world_size) {
    if (g_client.initialized) {
        return true;
    }
    int scale = 6;
    int width = (int)(world_size * scale);
    int height = (int)(world_size * scale);
    g_client.scale = scale;
    g_client.width = width;
    g_client.height = height;
    InitWindow(width, height, "Drone Swarm SAR");
    if (!IsWindowReady()) {
        fprintf(stderr, "raylib: failed to initialize window. Are you running headless?\n");
        g_client.initialized = false;
        return false;
    }
    SetTargetFPS(60);
    g_client.initialized = true;
    return true;
}

static void c_close_render(void) {
    if (!g_client.initialized) {
        return;
    }
    CloseWindow();
    g_client.initialized = false;
}

static void draw_obstacles(DroneSwarm *env) {
    if (env->obstacle_count == 0) {
        return;
    }
    Color fill = (Color){40, 70, 40, 255};
    Color outline = (Color){70, 110, 70, 255};
    int s = g_client.scale;
    for (int i = 0; i < env->obstacle_count; i++) {
        float *rect = env->obstacles[i];
        float x = rect[0] * s;
        float y = rect[1] * s;
        float w = (rect[2] - rect[0]) * s;
        float h = (rect[3] - rect[1]) * s;
        DrawRectangle((int)x, g_client.height - (int)(y + h), (int)w, (int)h, fill);
        DrawRectangleLines((int)x, g_client.height - (int)(y + h), (int)w, (int)h, outline);
    }
}

static void draw_comm_links(DroneSwarm *env) {
    int n = env->cfg.n_drones;
    float r_comm = env->cfg.r_comm;
    int s = g_client.scale;
    Color line = (Color){0, 150, 150, 100};
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dx = env->positions[i][0] - env->positions[j][0];
            float dy = env->positions[i][1] - env->positions[j][1];
            if (dx * dx + dy * dy <= r_comm * r_comm) {
                DrawLine(
                    (int)(env->positions[i][0] * s),
                    g_client.height - (int)(env->positions[i][1] * s),
                    (int)(env->positions[j][0] * s),
                    g_client.height - (int)(env->positions[j][1] * s),
                    line
                );
            }
        }
    }
}

static void draw_base(DroneSwarm *env) {
    int s = g_client.scale;
    Color base = (Color){255, 215, 0, 255};
    int bx = (int)(env->cfg.base_pos[0] * s);
    int by = g_client.height - (int)(env->cfg.base_pos[1] * s);
    DrawCircle(bx, by, 5, base);
    DrawCircleLines(bx, by, (int)(env->cfg.r_comm * s), (Color){255, 215, 0, 60});
}

static void draw_victims(DroneSwarm *env) {
    int s = g_client.scale;
    for (int i = 0; i < env->cfg.n_victims; i++) {
        int x = (int)(env->victim_pos[i][0] * s);
        int y = g_client.height - (int)(env->victim_pos[i][1] * s);
        Color color = (Color){100, 100, 100, 255};
        if (env->victim_status[i] == 1) {
            color = (Color){255, 165, 0, 255};
        } else if (env->victim_status[i] == 2) {
            color = (Color){0, 255, 0, 255};
        }
        DrawCircle(x, y, 4, color);
    }
}

static void draw_drones(DroneSwarm *env) {
    static Color colors[8] = {
        (Color){0, 187, 187, 255},
        (Color){187, 0, 187, 255},
        (Color){0, 187, 0, 255},
        (Color){187, 187, 0, 255},
        (Color){187, 0, 0, 255},
        (Color){0, 0, 187, 255},
        (Color){187, 127, 0, 255},
        (Color){127, 0, 187, 255},
    };
    int s = g_client.scale;
    for (int i = 0; i < env->cfg.n_drones; i++) {
        int x = (int)(env->positions[i][0] * s);
        int y = g_client.height - (int)(env->positions[i][1] * s);
        Color color = colors[i % 8];
        if (!env->connected[i]) {
            color = (Color){color.r, color.g, color.b, 120};
        }
        DrawCircle(x, y, 5, color);
    }
}

static void c_render(DroneSwarm *env) {
    if (!init_render_client(env->cfg.world_size)) {
        return;
    }
    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    draw_obstacles(env);
    draw_comm_links(env);
    draw_base(env);
    draw_victims(env);
    draw_drones(env);
    EndDrawing();
}
