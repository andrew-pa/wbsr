#include "cmmn.h"
#include "texture.h"
using namespace agt;

int main() {
	texture2d tex{ uvec2(640, 400) };

	tex.draw_text("wbsr", uvec2(8, 8), vec3(1.f, 1.f, 0.f));
	ostringstream filename; filename << "rndr" << time(nullptr) << ".bmp";
	tex.write_bmp(filename.str());
}