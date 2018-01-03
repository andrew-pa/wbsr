#include "cmmn.h"
#include "texture.h"
using namespace agt;

#include <glm/gtc/random.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

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

//#define AA
//#define DEBUG_SHADOW

void draw(const vector<rs_vertex>& vertices, const vector<uint32>& indices, function<void(uvec2, vec3, float, const rs_vertex&,const rs_vertex&,const rs_vertex&)> F) {
	for (int i = 0; i < indices.size(); i += 3) {
		const auto& v0 = vertices[indices[i]];
		const auto& v1 = vertices[indices[i+1]];
		const auto& v2 = vertices[indices[i+2]];

		// calculate screen bounds for triangle
		auto min_x = floor(glm::min(v0.pos.x, glm::min(v1.pos.x, v2.pos.x)));
		auto min_y = floor(glm::min(v0.pos.y, glm::min(v1.pos.y, v2.pos.y)));
		auto max_x = ceil(glm::max(v0.pos.x, glm::max(v1.pos.x, v2.pos.x)));
		auto max_y = ceil(glm::max(v0.pos.y, glm::max(v1.pos.y, v2.pos.y)));
		
		// inital barycentric coords (γ, α, β)
		vec3 gab = vec3(
			v0.pos.x*v1.pos.y - v1.pos.x*v0.pos.y,
			v1.pos.x*v2.pos.y - v2.pos.x*v1.pos.y,
			v2.pos.x*v0.pos.y - v0.pos.x*v2.pos.y);
		// change in (γ, α, β) with respect to x
		vec3 dgab_dx = vec3(
			v0.pos.y - v1.pos.y,
			v1.pos.y - v2.pos.y,
			v2.pos.y - v0.pos.y);
		// change in (γ, α, β) with respect to y
		vec3 dgab_dy = vec3(
			v1.pos.x - v0.pos.x,
			v2.pos.x - v1.pos.x,
			v0.pos.x - v2.pos.x);

		// calculate line equation values for the vertices
		float g0d = dgab_dx.x * v2.pos.x + dgab_dy.x * v2.pos.y + gab.x;
		float a0d = dgab_dx.y * v0.pos.x + dgab_dy.y * v0.pos.y + gab.y;
		float b0d = dgab_dx.z * v1.pos.x + dgab_dy.z * v1.pos.y + gab.z;

		// calculate line equation values for (-1, -1) to provide a tiebreaker for shared edges
		vec3 det = vec3(g0d, a0d, b0d) * (-dgab_dx + -dgab_dy + gab);

		// divide in the vertex line equations
		gab /= vec3(g0d, a0d, b0d);
		dgab_dx /= vec3(g0d, a0d, b0d);
		dgab_dy /= vec3(g0d, a0d, b0d);

		// increment (γ, α, β) to their initial values
		gab += dgab_dx * min_x + dgab_dy * min_y;

		vec3 zs = vec3(v2.pos.z, v0.pos.z, v1.pos.z);
		// partials of depth with respect to x,y
		float dz_dx = dot(zs, dgab_dx);
		float dz_dy = dot(zs, dgab_dy);
		// calculate initial depth at (min_x, min_y)
		float z = dot(zs, gab);

		// loop over screen bounds of the triangle
		for (ptrdiff_t y = min_y; y < max_y; ++y) {
			// store where this scanline started in (γ, α, β) and depth
			vec3 start_gab = gab; float start_z = z;
			for (ptrdiff_t x = min_x; x < max_x; ++x) {
#ifndef AA
				// is this pixel inside the triangle?
				if (gab.x >= 0 && gab.y >= 0 && gab.z >= 0) {
					// deal with shared edges
					if ((gab.x > 0 || det.x > 0) && (gab.y > 0 || det.y > 0) && (gab.z > 0 || det.z > 0)) {
						F(uvec2(x, y), gab, z, v0, v1, v2);
					}
				}
#else
				for (size_t su = 0; su < 2; ++su) {
					for (size_t sv = 0; sv < 2; ++sv) {
						float u = (su+linearRand(0.f, 1.f))/2.f;
						float v = (sv+linearRand(0.f, 1.f))/2.f;
						vec3 sgab = gab + dgab_dx * u + dgab_dy * v;
						// is this pixel inside the triangle?
						if (sgab.x >= 0 && sgab.y >= 0 && sgab.z >= 0) {
							// deal with shared edges
							if ((sgab.x > 0 || det.x > 0) && (sgab.y > 0 || det.y > 0) && (sgab.z > 0 || det.z > 0)) {
								F(uvec2(x, y), sgab, z + dz_dx*u + dz_dy*v, v0, v1, v2);
							}
						}
					}
				}
#endif
				gab += dgab_dx; z += dz_dx;
			}
			gab = start_gab + dgab_dy; z = start_z + dz_dy;
		}
	}
}

pair<vector<rs_vertex>, vector<uint32>> transform_clip(const vector<in_vertex>& vtc, const vector<uint32>& ixd, mat4 vp, mat4 world) {
	vector<rs_vertex> nvtc(vtc.size());
	// transform vertices from model → homogeneous projected space
	for (size_t i = 0; i < vtc.size(); ++i) {
		nvtc[i].posW = world * vec4(vtc[i].pos,1.f);
		nvtc[i].pos = vp * nvtc[i].posW;
		nvtc[i].pos.x /= nvtc[i].pos.w;
		nvtc[i].pos.y /= nvtc[i].pos.w;
		nvtc[i].pos.z /= nvtc[i].pos.w;
		nvtc[i].nor = (world * vec4(vtc[i].nor, 0.f)).xyz;
		nvtc[i].tex = vtc[i].tex;
	}

	/*vector<uint32> nixd(ixd.size());
	const float l = 0.f, r = 2048.f, b = 0.f, t = 2048.f, n = 0.1f, f = 10.f;
	for (size_t i = 0; i < ixd.size(); i += 3) {
		vec3 x = vec3(nvtc[ixd[i]].pos.x, nvtc[ixd[i+1]].pos.x, nvtc[ixd[i+2]].pos.x);
		vec3 y = vec3(nvtc[ixd[i]].pos.y, nvtc[ixd[i+1]].pos.y, nvtc[ixd[i+2]].pos.y);
		vec3 z = vec3(nvtc[ixd[i]].pos.z, nvtc[ixd[i+1]].pos.z, nvtc[ixd[i+2]].pos.z);
		vec3 w = vec3(nvtc[ixd[i]].pos.w, nvtc[ixd[i+1]].pos.w, nvtc[ixd[i+2]].pos.w);
		// clip against near plane
		vec3 np = -z + n * w;
		if (np.x > 0 || np.y > 0 || np.z > 0)
			continue;
		// clip against far plane
		vec3 fp = z - f * w;
		if (fp.x > 0 || fp.y > 0 || fp.z > 0)
			continue;


		nixd[i]   = ixd[i];
		nixd[i+1] = ixd[i+1];
		nixd[i+2] = ixd[i+2];
	}

	for (size_t i = 0; i < nvtc.size(); ++i) {
		nvtc[i].pos.x /= nvtc[i].pos.w;
		nvtc[i].pos.y /= nvtc[i].pos.w;
		nvtc[i].pos.z /= nvtc[i].pos.w;
	}*/

	return { nvtc, ixd };
}

mat4 window(uvec2 size) {
	return scale(translate(mat4(1), vec3((float)size.x / 2.f, (float)size.y / 2.f, 0.f)), vec3(size.x, -(float)size.y, 1.f));
}

int main() {
	auto start = chrono::system_clock::now();
	/* create render buffers */
	texture2d buf{
		//uvec2(320, 200)
		uvec2(640, 400)
		//uvec2(3840, 2160)
	};
	vector<float> depth(buf.size.x*buf.size.y, 1e9);
	const size_t shadow_size = 512;
	vector<float> shadow_depth(shadow_size*shadow_size, 1e9);

	/* initialize geometry */
	auto torus_vtc = vector<in_vertex>{ };
	auto torus_ixd = vector<uint32>{ };
	generate_torus(vec2(1.f, 0.5f), 32, [&torus_vtc](vec3 p, vec3 n, vec3, vec2 texcoord) {
		torus_vtc.push_back(in_vertex{ p, n, texcoord });
	}, [&torus_ixd](size_t i) {
		torus_ixd.push_back(i);
	});

	auto room_vtc = vector<in_vertex>{};
	auto room_ixd = vector<uint32>{};

	{
		tinyobj::attrib_t attrib;
		vector<tinyobj::shape_t> shapes; vector<tinyobj::material_t> materials;
		string err;
		tinyobj::LoadObj(&attrib, &shapes, &materials, &err, "teapot.obj");
		if (!err.empty()) {
			cout << "error: " << err << endl;
			getchar();
			return 1;
		}

		unordered_map<size_t, uint32> vertices;
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[0].mesh.num_face_vertices.size(); f++) {
			int fv = shapes[0].mesh.num_face_vertices[f];
			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				tinyobj::index_t idx = shapes[0].mesh.indices[index_offset + v];
				size_t hash = idx.vertex_index ^ idx.normal_index << 16 ^ idx.texcoord_index << 24;
				auto pov = vertices.find(hash);
				if (pov != vertices.end()) room_ixd.push_back(pov->second);
				else {
					room_ixd.push_back(room_vtc.size());
					vertices[hash] = room_vtc.size();
					tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
					tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
					tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
					tinyobj::real_t nx = attrib.normals[3 * idx.normal_index + 0];
					tinyobj::real_t ny = attrib.normals[3 * idx.normal_index + 1];
					tinyobj::real_t nz = attrib.normals[3 * idx.normal_index + 2];
					tinyobj::real_t tx = attrib.texcoords[2 * idx.texcoord_index + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * idx.texcoord_index + 1];
					room_vtc.push_back(in_vertex{ {vx,vy,vz},{nx,ny,nz},{tx,ty} });
				}
			}
			index_offset += fv;
		}
	}

	/* textures */
	checkerboard_texture tex{ vec3(0.2f), vec3(1.f), 16.f };

	/* object transforms */
	mat4 torus_W = rotate(translate(mat4(1), vec3(3.5f, 1.5f, 2.f)), pi<float>() / 6.f, vec3(1.f, 1.f, 0.f));
	mat4 room_W = scale(mat4(1), vec3(.9f));

	/* camera transforms */
	mat4 V = lookAt(vec3(-3.f, 6.0f, 8.f), vec3(1.f, 0.7f, 1.f), vec3(0.f, 1.f, 0.f));
	mat4 P = perspectiveFov(pi<float>() / 3.f, (float)buf.size.x, (float)buf.size.y, 0.1f, 10.f);
	mat4 Wn = window(buf.size);
	mat4  WnPV = Wn * P * V;

	auto start_transform = chrono::system_clock::now();

	/* transform geometry */
	auto torus_tf = transform_clip(torus_vtc, torus_ixd, WnPV, torus_W);
	auto room_tf = transform_clip(room_vtc, room_ixd, WnPV, room_W);

	/* deal with lights */
	const vec3 L = normalize(vec3(-0.5f, 1.f, -0.2f));

	/* compute light 'camera' transforms */
	mat4 lightV = lookAt(4.f*L, vec3(0.f), vec3(0.f, 1.f, 0.f));
	mat4 lightP = ortho(-12.f, 12.f, -12.f, 12.f, 0.5f, 8.f);
	mat4 lightWnPV = window(uvec2(shadow_size))*lightP*lightV;
	//	transform geometry wrt the light
	auto torus_light_tf = transform_clip(torus_vtc, torus_ixd, lightWnPV, torus_W);
	auto room_light_tf = transform_clip(room_vtc, room_ixd, lightWnPV, room_W);

	auto transform_raster = chrono::system_clock::now();

	/* compute shadow map */
	auto shadow = [&shadow_depth, &shadow_size](uvec2 px, vec3 gab, float z, const rs_vertex& v0, const rs_vertex& v1, const rs_vertex& v2) {
		if (px.x < 0 || px.x >= shadow_size || px.y < 0 || px.y >= shadow_size) return;
		if (z < shadow_depth[px.x + px.y*shadow_size]) {
			shadow_depth[px.x + px.y*shadow_size] = z;
		}
	};
	draw(torus_light_tf.first, torus_light_tf.second, shadow);
	draw(room_light_tf.first, room_light_tf.second, shadow);

	/* debug output of shadow map */
#ifdef DEBUG_SHADOW
	texture2d shadow_map{ uvec2(shadow_size) };
	for (size_t y = 0; y < shadow_size; ++y)
		for (size_t x = 0; x < shadow_size; ++x) {
			float z = shadow_depth[x + y * shadow_size];
			shadow_map.pixel(uvec2(x, y)) = vec3(z > 1000.f ? 0.f : z);
		}
	shadow_map.write_bmp("shadow.bmp");
#endif

	auto raster_shadow = chrono::system_clock::now();

	/* rasterize image */
	auto shade = [&](uvec2 px, vec3 gab, float z, const rs_vertex& v0, const rs_vertex& v1, const rs_vertex& v2) {
		if (px.x < 0 || px.x >= buf.size.x || px.y < 0 || px.y >= buf.size.y) return;
		if (z <= depth[px.x + px.y * buf.size.x]) {
			// compute persepective correct interpolators
			float d = v1.pos.w*v2.pos.w + v2.pos.w*gab.z*(v0.pos.w - v1.pos.w) + v1.pos.w*gab.x*(v0.pos.w - v2.pos.w);
			float bw = v0.pos.w*v2.pos.w*gab.z / d;
			float gw = v0.pos.w*v1.pos.w*gab.x / d;
			float aw = 1.0f - bw - gw;

			// project this point as seen from light
			float shadow = 1.f;
			vec4 posW = aw * v0.posW + bw * v1.posW + gw * v2.posW;
			vec4 pos_light = lightWnPV * posW; // this would be much faster done in transform() but it would make transform() more specific
			pos_light /= pos_light.w;
			pos_light.xy = floor(pos_light.xy());
			if (pos_light.x >= 0 && pos_light.x < shadow_size && pos_light.y >= 0 && pos_light.y < shadow_size) {
				float lightZ = shadow_depth[pos_light.x + pos_light.y * shadow_size];
				if (abs(lightZ - pos_light.z) > 0.1f) shadow = 0.f;
			}
			//else shadow = 0.f;

			// interpolate texture coords and normals
			vec2 uv = aw * v0.tex + bw * v1.tex + gw * v2.tex;
			vec3 nor = aw * v0.nor + bw * v1.nor + gw * v2.nor;

			// compute shading
			vec3 col = vec3(0.9f, 0.8f, 0.6f)*glm::max(0.f, dot(nor, L))*shadow + vec3(0.f, 0.04f, 0.09f);

			// write to render target + gamma correction
#ifdef AA
			buf.pixel(px) += pow(col * tex.texel(uv), vec3(0.4545)) * 0.25f;
#else
			buf.pixel(px) = pow(col * tex.texel(uv), vec3(0.4545));
#endif
			depth[px.x + px.y * buf.size.x] = z;
		}
	};
	draw(torus_tf.first, torus_tf.second, shade);
	draw(room_tf.first, room_tf.second, shade);

	auto raster_end = chrono::system_clock::now();

	/* write render to file */
	ostringstream wm;
	wm << "total " << chrono::duration_cast<chrono::milliseconds>(raster_end - start).count() << "ms" << endl;
	wm << " init " << chrono::duration_cast<chrono::milliseconds>(start_transform - start).count() << "ms" << endl;
	wm << " trns " << chrono::duration_cast<chrono::milliseconds>(transform_raster - start_transform).count() << "ms" << endl;
	wm << " rstr " << chrono::duration_cast<chrono::milliseconds>(raster_end - transform_raster).count() << "ms" << endl;
	wm << "  shdw " << chrono::duration_cast<chrono::milliseconds>(raster_shadow - transform_raster).count() << "ms" << endl;
	wm << "  colr " << chrono::duration_cast<chrono::milliseconds>(raster_end - raster_shadow).count() << "ms" << endl;

	buf.draw_text(wm.str(), uvec2(8, 8), vec3(1.f, 1.f, 0.f));
	ostringstream filename; filename << "rndr" << time(nullptr) << ".bmp";
	buf.write_bmp(filename.str());
}