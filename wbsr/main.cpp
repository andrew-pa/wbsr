#include "cmmn.h"
#include "texture.h"
using namespace agt;

struct vertex {
	vec4 pos;
	vec3 col;
};

void draw(const vector<vertex>& vertices, const vector<uint32>& indices, texture2d& target) {
	for (int i = 0; i < indices.size(); i += 3) {
		auto& v0 = vertices[indices[i]];
		auto& v1 = vertices[indices[i+1]];
		auto& v2 = vertices[indices[i+2]];

		auto min_x = glm::min(v0.pos.x, glm::min(v1.pos.x, v2.pos.x));
		auto min_y = glm::min(v0.pos.y, glm::min(v1.pos.y, v2.pos.y));
		auto max_x = glm::max(v0.pos.x, glm::max(v1.pos.x, v2.pos.x));
		auto max_y = glm::max(v0.pos.y, glm::max(v1.pos.y, v2.pos.y));
		vec3 gab = vec3(
			v0.pos.x*v1.pos.y - v1.pos.x*v0.pos.y,
			v1.pos.x*v2.pos.y - v2.pos.x*v1.pos.y,
			v2.pos.x*v0.pos.y - v0.pos.x*v2.pos.y);
		vec3 dgab_dx = vec3(
			v0.pos.y - v1.pos.y,
			v1.pos.y - v2.pos.y,
			v2.pos.y - v0.pos.y);
		vec3 dgab_dy = vec3(
			v1.pos.x - v0.pos.x,
			v2.pos.x - v1.pos.x,
			v0.pos.x - v2.pos.x);

		float g0d = dgab_dx.x * v2.pos.x + dgab_dy.x * v2.pos.y + gab.x;
		float a0d = dgab_dx.y * v0.pos.x + dgab_dy.y * v0.pos.y + gab.y;
		float b0d = dgab_dx.z * v1.pos.x + dgab_dy.z * v1.pos.y + gab.z;
		gab /= vec3(g0d, a0d, b0d);
		dgab_dx /= vec3(g0d, a0d, b0d);
		dgab_dy /= vec3(g0d, a0d, b0d);
		gab += dgab_dx * min_x + dgab_dy * min_y;
		for (size_t y = min_y; y < max_y; ++y) {
			vec3 start_gab = gab;
			for (size_t x = min_x; x < max_x; ++x) {
				if(gab.x > 0 && gab.y > 0 && gab.z > 0)
					target.pixel(uvec2(x, y)) = gab.y * v0.col + gab.z * v1.col + gab.x * v2.col;
				gab += dgab_dx;
			}
			gab = start_gab + dgab_dy;
		}
	}
}

int main() {
	texture2d tex{ uvec2(640, 400) };

	auto vtc = vector<vertex>{
		{vec4(16.f, 16.f, 0.f, 1.f), vec3(1.f, 0.f, 0.f)},
		{vec4(16.f, 128.f, 0.f, 1.f), vec3(0.f, 1.f, 0.f)},
		{vec4(128.f, 16.f, 0.f, 1.f), vec3(1.f, 1.f, 1.f)},
		{vec4(128.f, 128.f, 0.f, 1.f), vec3(0.f, 0.f, 1.f)},
	};
	auto ixd = vector<uint32>{
		0,1,2,
		//1,2,3
	};

	draw(vtc, ixd, tex);

	tex.draw_text("wbsr", uvec2(8, 8), vec3(1.f, 1.f, 0.f));
	ostringstream filename; filename << "rndr" << time(nullptr) << ".bmp";
	tex.write_bmp(filename.str());
}