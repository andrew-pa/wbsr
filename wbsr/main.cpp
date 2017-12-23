#include "cmmn.h"
#include "texture.h"
using namespace agt;

struct vertex {
	vec4 pos;
	vec3 col;
};


void generate_torus(vec2 r, int div, function<void(vec3, vec3, vec3, vec2)> vertex, function<void(size_t)> index)
{
	int ring_count = div;
	int stack_count = div;

	vector<tuple<vec3,vec3,vec3,vec2>> frvtx;
	for (int i = 0; i < div + 1; ++i)
	{
		vec4 p = vec4(r.y, 0.f, 0.f, 1.f)*rotate(mat4(1), radians(i*360.f / (float)div), vec3(0, 0, 1))
			+ vec4(r.x, 0.f, 0.f, 1.f);
		vec2 tx = vec2(0, (float)i / div);
		vec4 tg = vec4(0.f, -1.f, 0.f, 1.f)*rotate(mat4(1), radians(i*360.f / (float)div), vec3(0, 0, 1));
		vec3 n = cross(vec3(tg), vec3(0.f, 0.f, -1.f));
		vertex(vec3(p), vec3(n), vec3(tg), tx);
		frvtx.push_back({vec3(p), n, vec3(tg), tx});
	}

	for (int ring = 1; ring < ring_count + 1; ++ring)
	{
		mat4 rot = rotate(mat4(1), radians(ring*360.f / (float)div), vec3(0, 1, 0));
		for (int i = 0; i < stack_count + 1; ++i)
		{
			vec4 p = vec4(get<0>(frvtx[i]), 1.f);
			vec4 nr = vec4(get<1>(frvtx[i]), 0.f);
			vec4 tg = vec4(get<2>(frvtx[i]), 0.f);
			p = p*rot;
			nr = nr*rot;
			tg = tg*rot;

			vertex(vec3(p), vec3(nr),
				vec3(tg), vec2(2.f*ring / (float)div, get<3>(frvtx[i]).y));
		}
	}

	for (int ring = 0; ring < ring_count; ++ring)
	{
		for (int i = 0; i < stack_count; ++i)
		{
			index(ring*(div + 1) + i);
			index((ring + 1)*(div + 1) + i);
			index(ring*(div + 1) + i + 1);

			index(ring*(div + 1) + i + 1);
			index((ring + 1)*(div + 1) + i);
			index((ring + 1)*(div + 1) + i + 1);
		}
	}
}

void generate_sphere(float radius, uint slice_count, uint stack_count, function<void(vec3,vec3,vec3,vec2)> vertex, function<void(uint32_t)> index)
{
	vertex(vec3(0.f, radius, 0.f), vec3(0, 1, 0), vec3(1, 0, 0), vec2(0, 0));

	float dphi = pi<float>() / stack_count;
	float dtheta = 2.f*pi<float>() / slice_count;

	for (uint i = 1; i <= stack_count - 1; ++i)
	{
		float phi = i*dphi;
		for (uint j = 0; j <= slice_count; ++j)
		{
			float theta = j*dtheta;
			vec3 p = vec3(radius*sinf(phi)*cosf(theta),
				radius*cosf(phi),
				radius*sinf(phi)*sinf(theta));
			vec3 t = normalize(vec3(-radius*sinf(phi)*sinf(theta),
				0.f,
				radius*sinf(phi)*cosf(theta)));
			vertex(p, normalize(p), t, vec2(theta / (2.f*pi<float>()), phi / (2.f*pi<float>())));

		}
	}

	vertex(vec3(0.f, -radius, 0.f), vec3(0, -1, 0), vec3(1, 0, 0), vec2(0, 1));

	for (uint32_t i = 1; i <= slice_count; ++i)
	{
		index(0);
		index(i + 1);
		index(i);
	}

	uint32_t bi = 1;
	uint32_t rvc = slice_count + 1;
	for (uint32_t i = 0; i < stack_count - 2; ++i)
	{
		for (uint j = 0; j < slice_count; ++j)
		{
			index(bi + i*rvc + j);
			index(bi + i*rvc + j + 1);
			index(bi + (i + 1)*rvc + j);
			index(bi + (i + 1)*rvc + j);
			index(bi + (i*rvc + j + 1));
			index(bi + (i + 1)*rvc + j + 1);
		}
	}

	uint32_t spi = (uint32_t)(1+(stack_count-1)*slice_count);
	bi = spi - rvc;
	for (uint i = 0; i < slice_count; ++i)
	{
		index(spi);
		index(bi + i);
		index(bi + i + 1);
	}
}

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

void transform(vector<vertex>& vtc, mat4 transform) {
	for (auto& v : vtc) {
		v.pos = transform * v.pos;
		v.pos /= v.pos.w;
	}
}

int main() {
	texture2d tex{ uvec2(640, 400) };

	auto vtc = vector<vertex>{
		/*{vec4(0.f, 0.f, 0.f, 1.f), vec3(1.f, 0.f, 0.f)},
		{vec4(0.f, 1.f, 0.f, 1.f), vec3(0.f, 1.f, 0.f)},
		{vec4(1.f, 0.f, 0.f, 1.f), vec3(0.f, 0.f, 1.f)},
		{vec4(1.f, 1.f, 0.f, 1.f), vec3(1.f, 1.f, 1.f)},*/
	};
	auto ixd = vector<uint32>{
		/*0,1,2,
		1,2,3*/
	};

	const vec3 L = normalize(vec3(0.5f, 1.f, 0.2f));

	generate_sphere(1.f, 8, 8, [&vtc, L](vec3 p, vec3 n, vec3, vec2) {
		vtc.push_back(vertex{ vec4(p, 1.f), vec3(0.8f, 0.6f, 0.f)*glm::max(0.f, dot(n, L))+vec3(0.f, 0.05f, 0.15f) });
	}, [&ixd](size_t i) {
		ixd.push_back(i);
	});

	// screen space = [0,0,0] .. [640, 400, 1]
	// - projection matrix -
	// view space 
	// - view matrix -
	// world space
	// - world matrix -
	// model space

	mat4 V = lookAt(vec3(0.f, 0.f, 8.f), vec3(0.f), vec3(0.f, 1.f, 0.f));
	mat4 P = perspectiveFov(pi<float>() / 4.f, 640.f, 400.f, 0.01f, 100.f);
	mat4 Wn = scale(translate(mat4(1), vec3(320.f, 200.f, 0.f)), vec3(640.f, -400.f, 1.f));
	transform(vtc, Wn*P*V);//rotate(translate(mat4(1), vec3(320.f, 200.f, 0.f)), 1.f, vec3(0.f, 0.f, 1.f)));

	draw(vtc, ixd, tex);

	tex.draw_text("wbsr", uvec2(8, 8), vec3(1.f, 1.f, 0.f));
	ostringstream filename; filename << "rndr" << time(nullptr) << ".bmp";
	tex.write_bmp(filename.str());
}