#include "cmmn.h"
#include "texture.h"
using namespace agt;

struct in_vertex {
	vec3 pos;
	vec3 nor;
	vec2 tex;

	in_vertex() {}
	in_vertex(vec3 p, vec3 n, vec2 t)
		: pos(p), nor(n), tex(t) {}
};

struct rs_vertex {
	vec4 pos;
	vec4 posW;
	vec3 nor;
	vec2 tex;

	rs_vertex() {}
	rs_vertex(vec4 p, vec4 pW, vec3 n, vec2 t)
		: pos(p), posW(pW), nor(n), tex(t) {}
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

void draw(const vector<rs_vertex>& vertices, const vector<uint32>& indices, function<void(uvec2, vec3, float, const rs_vertex&,const rs_vertex&,const rs_vertex&)> F) {
	for (int i = 0; i < indices.size(); i += 3) {
		const auto& v0 = vertices[indices[i]];
		const auto& v1 = vertices[indices[i+1]];
		const auto& v2 = vertices[indices[i+2]];

		auto min_x = floor(glm::min(v0.pos.x, glm::min(v1.pos.x, v2.pos.x)));
		auto min_y = floor(glm::min(v0.pos.y, glm::min(v1.pos.y, v2.pos.y)));
		auto max_x = ceil(glm::max(v0.pos.x, glm::max(v1.pos.x, v2.pos.x)));
		auto max_y = ceil(glm::max(v0.pos.y, glm::max(v1.pos.y, v2.pos.y)));
		
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
		vec3 det = vec3(g0d, a0d, b0d) * (-dgab_dx + -dgab_dy + gab);
		gab /= vec3(g0d, a0d, b0d);
		dgab_dx /= vec3(g0d, a0d, b0d);
		dgab_dy /= vec3(g0d, a0d, b0d);
		gab += dgab_dx * min_x + dgab_dy * min_y;

		vec3 zs = vec3(v2.pos.z, v0.pos.z, v1.pos.z);
		float dz_dx = dot(zs, dgab_dx);
		float dz_dy = dot(zs, dgab_dy);
		float z = dot(zs, gab);

		for (ptrdiff_t y = min_y; y < max_y; ++y) {
			vec3 start_gab = gab; float start_z = z;
			for (ptrdiff_t x = min_x; x < max_x; ++x) {
				if (gab.x >= 0 && gab.y >= 0 && gab.z >= 0) {
					if ((gab.x > 0 || det.x > 0) && (gab.y > 0 || det.y > 0) && (gab.z > 0 || det.z > 0)) {
						F(uvec2(x, y), gab, z, v0, v1, v2);
					}
				}
				gab += dgab_dx; z += dz_dx;
			}
			gab = start_gab + dgab_dy; z = start_z + dz_dy;
		}
	}
}

vector<rs_vertex> transform(const vector<in_vertex>& vtc, mat4 vp, mat4 world) {
	vector<rs_vertex> nvtc(vtc.size());
	for (size_t i = 0; i < vtc.size(); ++i) {
		nvtc[i].posW = world * vec4(vtc[i].pos,1.f);
		nvtc[i].pos = vp * nvtc[i].posW;
		nvtc[i].pos.x /= nvtc[i].pos.w;
		nvtc[i].pos.y /= nvtc[i].pos.w;
		nvtc[i].pos.z /= nvtc[i].pos.w;
		nvtc[i].nor = (world * vec4(vtc[i].nor, 0.f)).xyz;
		nvtc[i].tex = vtc[i].tex;
	}
	return nvtc;
}

mat4 window(uvec2 size) {
	return scale(translate(mat4(1), vec3((float)size.x / 2.f, (float)size.y / 2.f, 0.f)), vec3(size.x, -(float)size.y, 1.f));
}

int main() {
	/* create render buffers */
	texture2d buf{
		uvec2(640, 400)
		//uvec2(3840, 2160)
	};
	vector<float> depth(buf.size.x*buf.size.y, 1e9);
	const size_t shadow_size = 2048;
	vector<float> shadow_depth(shadow_size*shadow_size, 1e9);

	/* initialize geometry */
	auto torus_vtc = vector<in_vertex>{ };
	auto torus_ixd = vector<uint32>{ };
	generate_torus(vec2(1.f, 0.5f), 32, [&torus_vtc](vec3 p, vec3 n, vec3, vec2 texcoord) {
		torus_vtc.push_back(in_vertex{ p, n, texcoord });
	}, [&torus_ixd](size_t i) {
		torus_ixd.push_back(i);
	});

	auto floor_vtc = vector<in_vertex>{
		{vec3(-1.f, 0.f, -1.f), vec3(0.f, 1.f, 0.f), vec2(0.f)},
		{vec3(-1.f, 0.f, 1.f), vec3(0.f, 1.f, 0.f), vec2(0.f, 1.f)},
		{vec3(1.f, 0.f, -1.f), vec3(0.f, 1.f, 0.f), vec2(1.f, 0.f)},
		{vec3(1.f, 0.f, 1.f), vec3(0.f, 1.f, 0.f), vec2(1.f)},
	};
	auto floor_ixd = vector<uint32>{ 0, 1, 2, 1, 2, 3 };

	/* textures */
	checkerboard_texture tex{ vec3(0.2f), vec3(1.f), 16.f };

	/* object transforms */
	mat4 torus_W = mat4(1);// rotate(mat4(1), pi<float>() / 3.f, vec3(1.f, 1.f, 0.f));
	mat4 floor_W = scale(translate(mat4(1), vec3(0.f, -1.5f, 0.f)), vec3(3.f));

	/* camera transforms */
	mat4 V = lookAt(vec3(-3.f, 4.f, 12.f), vec3(0.f, -0.5f, 0.f), vec3(0.f, 1.f, 0.f));
	mat4 P = perspectiveFov(pi<float>() / 4.f, (float)buf.size.x, (float)buf.size.y, 0.1f, 10.f);
	mat4 Wn = window(buf.size);
	mat4  WnPV = Wn * P * V;

	/* transform geometry */
	auto torus_tvtc = transform(torus_vtc, WnPV, torus_W);
	auto floor_tvtc = transform(floor_vtc, WnPV, floor_W);

	/* deal with lights */
	const vec3 L = normalize(vec3(-0.5f, 1.f, -0.2f));

	/* compute light 'camera' transforms */
	mat4 lightV = lookAt(8.f*L, vec3(0.f), vec3(0.f, 1.f, 0.f));
	mat4 lightP =// perspectiveFov(pi<float>() / 3.f, (float)shadow_size, (float)shadow_size, 0.1f, 16.f);
					ortho(-6.f, 6.f, -6.f, 6.f, 0.1f, 10.f);
	mat4 lightWnPV = window(uvec2(shadow_size))*lightP*lightV;
	//	transform geometry wrt the light
	auto torus_light_tvtc = transform(torus_vtc, lightWnPV, torus_W);
	auto floor_light_tvtc = transform(floor_vtc, lightWnPV, floor_W);

	/* compute shadow map */
	auto shadow = [&shadow_depth, &shadow_size](uvec2 px, vec3 gab, float z, const rs_vertex& v0, const rs_vertex& v1, const rs_vertex& v2) {
		if (px.x < 0 || px.x >= shadow_size || px.y < 0 || px.y >= shadow_size) return;
		if (z < shadow_depth[px.x + px.y*shadow_size]) {
			shadow_depth[px.x + px.y*shadow_size] = z;
		}
	};
	draw(torus_light_tvtc, torus_ixd, shadow);
	draw(floor_light_tvtc, floor_ixd, shadow);

	/* debug output of shadow map */
	texture2d shadow_map{ uvec2(shadow_size) };
	for (size_t y = 0; y < shadow_size; ++y)
		for (size_t x = 0; x < shadow_size; ++x) {
			float z = shadow_depth[x + y * shadow_size];
			shadow_map.pixel(uvec2(x, y)) = vec3(z > 1000.f ? 0.f : z);
		}
	shadow_map.write_bmp("shadow.bmp");

	/* rasterize image */
	auto shade = [&](uvec2 px, vec3 gab, float z, const rs_vertex& v0, const rs_vertex& v1, const rs_vertex& v2) {
		if (px.x < 0 || px.x >= buf.size.x || px.y < 0 || px.y >= buf.size.y) return;
		if (z < depth[px.x + px.y * buf.size.x]) {
			// compute persepective correct interpolators
			float d = v1.pos.w*v2.pos.w + v2.pos.w*gab.z*(v0.pos.w - v1.pos.w) + v1.pos.w*gab.x*(v0.pos.w - v2.pos.w);
			float bw = v0.pos.w*v2.pos.w*gab.z / d;
			float gw = v0.pos.w*v1.pos.w*gab.x / d;
			float aw = 1.0f - bw - gw;

			// project this point as seen from light
			vec4 posW = aw * v0.posW + bw * v1.posW + gw * v2.posW;
			vec4 pos_light = lightWnPV * posW;
			pos_light /= pos_light.w;
			pos_light.xy = floor(pos_light.xy());
			float shadow = 1.f;
			if (pos_light.x >= 0 && pos_light.x < shadow_size && pos_light.y >= 0 && pos_light.y < shadow_size) {
				float lightZ = shadow_depth[pos_light.x + pos_light.y * shadow_size];
				if (abs(lightZ - pos_light.z) > 0.001f) shadow = 0.f;
			}
			//else shadow = 0.f;

			// interpolate texture coords and normals
			vec2 uv = aw * v0.tex + bw * v1.tex + gw * v2.tex;
			vec3 nor = aw * v0.nor + bw * v1.nor + gw * v2.nor;

			// compute shading
			vec3 col = vec3(0.9f, 0.8f, 0.6f)*glm::max(0.f, dot(nor, L))*shadow + vec3(0.f, 0.04f, 0.09f);

			// write to render target + gamma correction
			buf.pixel(px) = pow(col * tex.texel(uv), vec3(0.4545));
			depth[px.x + px.y * buf.size.x] = z;
		}
	};
	draw(torus_tvtc, torus_ixd, shade);
	draw(floor_tvtc, floor_ixd, shade);

	/* write render to file */
	buf.draw_text("wbsr", uvec2(8, 8), vec3(1.f, 1.f, 0.f));
	ostringstream filename; filename << "rndr" << time(nullptr) << ".bmp";
	buf.write_bmp(filename.str());
}