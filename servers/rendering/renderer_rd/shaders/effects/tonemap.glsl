#[vertex]

#version 450

#VERSION_DEFINES

#ifdef MULTIVIEW
#ifdef has_VK_KHR_multiview
#extension GL_EXT_multiview : enable
#endif
#endif

layout(location = 0) out vec2 uv_interp;

void main() {
	vec2 base_arr[4] = vec2[](vec2(0.0, 0.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(1.0, 0.0));
	uv_interp = base_arr[gl_VertexIndex];
	gl_Position = vec4(uv_interp * 2.0 - 1.0, 0.0, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#ifdef MULTIVIEW
#ifdef has_VK_KHR_multiview
#extension GL_EXT_multiview : enable
#define ViewIndex gl_ViewIndex
#else // has_VK_KHR_multiview
#define ViewIndex 0
#endif // has_VK_KHR_multiview
#endif //MULTIVIEW

layout(location = 0) in vec2 uv_interp;

#ifdef SUBPASS
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput input_color;
#elif defined(MULTIVIEW)
layout(set = 0, binding = 0) uniform sampler2DArray source_color;
#else
layout(set = 0, binding = 0) uniform sampler2D source_color;
#endif

layout(set = 1, binding = 0) uniform sampler2D source_auto_exposure;
#ifdef MULTIVIEW
layout(set = 2, binding = 0) uniform sampler2DArray source_glow;
#else
layout(set = 2, binding = 0) uniform sampler2D source_glow;
#endif
layout(set = 2, binding = 1) uniform sampler2D glow_map;

#ifdef USE_1D_LUT
layout(set = 3, binding = 0) uniform sampler2D source_color_correction;
#else
layout(set = 3, binding = 0) uniform sampler3D source_color_correction;
#endif

layout(push_constant, std430) uniform Params {
	vec3 bcs;
	bool use_bcs;

	bool use_glow;
	bool use_auto_exposure;
	bool use_color_correction;
	uint tonemapper;

	uvec2 glow_texture_size;
	float glow_intensity;
	float glow_map_strength;

	uint glow_mode;
	float glow_levels[7];

	float exposure;
	float white;
	float auto_exposure_scale;
	float luminance_multiplier;

	vec2 pixel_size;
	bool use_fxaa;
	bool use_debanding;
}
params;

layout(location = 0) out vec4 frag_color;

#ifdef USE_GLOW_FILTER_BICUBIC
// w0, w1, w2, and w3 are the four cubic B-spline basis functions
float w0(float a) {
	return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a) {
	return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a) {
	return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a) {
	return (1.0f / 6.0f) * (a * a * a);
}

// g0 and g1 are the two amplitude functions
float g0(float a) {
	return w0(a) + w1(a);
}

float g1(float a) {
	return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
float h0(float a) {
	return -1.0f + w1(a) / (w0(a) + w1(a));
}

float h1(float a) {
	return 1.0f + w3(a) / (w2(a) + w3(a));
}

#ifdef MULTIVIEW
vec4 texture2D_bicubic(sampler2DArray tex, vec2 uv, int p_lod) {
	float lod = float(p_lod);
	vec2 tex_size = vec2(params.glow_texture_size >> p_lod);
	vec2 pixel_size = vec2(1.0f) / tex_size;

	uv = uv * tex_size + vec2(0.5f);

	vec2 iuv = floor(uv);
	vec2 fuv = fract(uv);

	float g0x = g0(fuv.x);
	float g1x = g1(fuv.x);
	float h0x = h0(fuv.x);
	float h1x = h1(fuv.x);
	float h0y = h0(fuv.y);
	float h1y = h1(fuv.y);

	vec3 p0 = vec3((vec2(iuv.x + h0x, iuv.y + h0y) - vec2(0.5f)) * pixel_size, ViewIndex);
	vec3 p1 = vec3((vec2(iuv.x + h1x, iuv.y + h0y) - vec2(0.5f)) * pixel_size, ViewIndex);
	vec3 p2 = vec3((vec2(iuv.x + h0x, iuv.y + h1y) - vec2(0.5f)) * pixel_size, ViewIndex);
	vec3 p3 = vec3((vec2(iuv.x + h1x, iuv.y + h1y) - vec2(0.5f)) * pixel_size, ViewIndex);

	return (g0(fuv.y) * (g0x * textureLod(tex, p0, lod) + g1x * textureLod(tex, p1, lod))) +
			(g1(fuv.y) * (g0x * textureLod(tex, p2, lod) + g1x * textureLod(tex, p3, lod)));
}

#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) texture2D_bicubic(m_tex, m_uv, m_lod)
#else // MULTIVIEW

vec4 texture2D_bicubic(sampler2D tex, vec2 uv, int p_lod) {
	float lod = float(p_lod);
	vec2 tex_size = vec2(params.glow_texture_size >> p_lod);
	vec2 pixel_size = vec2(1.0f) / tex_size;

	uv = uv * tex_size + vec2(0.5f);

	vec2 iuv = floor(uv);
	vec2 fuv = fract(uv);

	float g0x = g0(fuv.x);
	float g1x = g1(fuv.x);
	float h0x = h0(fuv.x);
	float h1x = h1(fuv.x);
	float h0y = h0(fuv.y);
	float h1y = h1(fuv.y);

	vec2 p0 = (vec2(iuv.x + h0x, iuv.y + h0y) - vec2(0.5f)) * pixel_size;
	vec2 p1 = (vec2(iuv.x + h1x, iuv.y + h0y) - vec2(0.5f)) * pixel_size;
	vec2 p2 = (vec2(iuv.x + h0x, iuv.y + h1y) - vec2(0.5f)) * pixel_size;
	vec2 p3 = (vec2(iuv.x + h1x, iuv.y + h1y) - vec2(0.5f)) * pixel_size;

	return (g0(fuv.y) * (g0x * textureLod(tex, p0, lod) + g1x * textureLod(tex, p1, lod))) +
			(g1(fuv.y) * (g0x * textureLod(tex, p2, lod) + g1x * textureLod(tex, p3, lod)));
}

#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) texture2D_bicubic(m_tex, m_uv, m_lod)
#endif // !MULTIVIEW

#else // USE_GLOW_FILTER_BICUBIC

#ifdef MULTIVIEW
#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) textureLod(m_tex, vec3(m_uv, ViewIndex), float(m_lod))
#else // MULTIVIEW
#define GLOW_TEXTURE_SAMPLE(m_tex, m_uv, m_lod) textureLod(m_tex, m_uv, float(m_lod))
#endif // !MULTIVIEW

#endif // !USE_GLOW_FILTER_BICUBIC









#define M_PI 3.1415926535897932384626433832795

float cbrt( float x )
{
    return sign(x)*pow(abs(x),1.0f/3.0f);
}

float srgb_transfer_function(float a)
{
	return .0031308f >= a ? 12.92f * a : 1.055f * pow(a, .4166666666666667f) - .055f;
}

float srgb_transfer_function_inv(float a)
{
	return .04045f < a ? pow((a + .055f) / 1.055f, 2.4f) : a / 12.92f;
}

vec3 linear_srgb_to_oklab(vec3 c)
{
	float l = 0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b;
	float m = 0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b;
	float s = 0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b;

	float l_ = cbrt(l);
	float m_ = cbrt(m);
	float s_ = cbrt(s);

	return vec3(
		0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_,
		1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_,
		0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_
	);
}

vec3 oklab_to_linear_srgb(vec3 c)
{
	float l_ = c.x + 0.3963377774f * c.y + 0.2158037573f * c.z;
	float m_ = c.x - 0.1055613458f * c.y - 0.0638541728f * c.z;
	float s_ = c.x - 0.0894841775f * c.y - 1.2914855480f * c.z;

	float l = l_ * l_ * l_;
	float m = m_ * m_ * m_;
	float s = s_ * s_ * s_;

	return vec3(
		+4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s,
		-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s,
		-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s
	);
}

// Finds the maximum saturation possible for a given hue that fits in sRGB
// Saturation here is defined as S = C/L
// a and b must be normalized so a^2 + b^2 == 1
float compute_max_saturation(float a, float b)
{
	// Max saturation will be when one of r, g or b goes below zero.

	// Select different coefficients depending on which component goes below zero first
	float k0, k1, k2, k3, k4, wl, wm, ws;

	if (-1.88170328f * a - 0.80936493f * b > 1.f)
	{
		// Red component
		k0 = +1.19086277f; k1 = +1.76576728f; k2 = +0.59662641f; k3 = +0.75515197f; k4 = +0.56771245f;
		wl = +4.0767416621f; wm = -3.3077115913f; ws = +0.2309699292f;
	}
	else if (1.81444104f * a - 1.19445276f * b > 1.f)
	{
		// Green component
		k0 = +0.73956515f; k1 = -0.45954404f; k2 = +0.08285427f; k3 = +0.12541070f; k4 = +0.14503204f;
		wl = -1.2684380046f; wm = +2.6097574011f; ws = -0.3413193965f;
	}
	else
	{
		// Blue component
		k0 = +1.35733652f; k1 = -0.00915799f; k2 = -1.15130210f; k3 = -0.50559606f; k4 = +0.00692167f;
		wl = -0.0041960863f; wm = -0.7034186147f; ws = +1.7076147010f;
	}

	// Approximate max saturation using a polynomial:
	float S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b;

	// Do one step Halley's method to get closer
	// this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
	// this should be sufficient for most applications, otherwise do two/three steps 

	float k_l = +0.3963377774f * a + 0.2158037573f * b;
	float k_m = -0.1055613458f * a - 0.0638541728f * b;
	float k_s = -0.0894841775f * a - 1.2914855480f * b;

	{
		float l_ = 1.f + S * k_l;
		float m_ = 1.f + S * k_m;
		float s_ = 1.f + S * k_s;

		float l = l_ * l_ * l_;
		float m = m_ * m_ * m_;
		float s = s_ * s_ * s_;

		float l_dS = 3.f * k_l * l_ * l_;
		float m_dS = 3.f * k_m * m_ * m_;
		float s_dS = 3.f * k_s * s_ * s_;

		float l_dS2 = 6.f * k_l * k_l * l_;
		float m_dS2 = 6.f * k_m * k_m * m_;
		float s_dS2 = 6.f * k_s * k_s * s_;

		float f = wl * l + wm * m + ws * s;
		float f1 = wl * l_dS + wm * m_dS + ws * s_dS;
		float f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2;

		S = S - f * f1 / (f1 * f1 - 0.5f * f * f2);
	}

	return S;
}

// finds L_cusp and C_cusp for a given hue
// a and b must be normalized so a^2 + b^2 == 1
vec2 find_cusp(float a, float b)
{
	// First, find the maximum saturation (saturation S = C/L)
	float S_cusp = compute_max_saturation(a, b);

	// Convert to linear sRGB to find the first point where at least one of r,g or b >= 1:
	vec3 rgb_at_max = oklab_to_linear_srgb(vec3( 1, S_cusp * a, S_cusp * b ));
	float L_cusp = cbrt(1.f / max(max(rgb_at_max.r, rgb_at_max.g), rgb_at_max.b));
	float C_cusp = L_cusp * S_cusp;

	return vec2( L_cusp , C_cusp );
}
// Finds intersection of the line defined by 
// L = L0 * (1 - t) + t * L1;
// C = t * C1;
// a and b must be normalized so a^2 + b^2 == 1
float find_gamut_intersection(float a, float b, float L1, float C1, float L0, vec2 cusp)
{
	// Find the intersection for upper and lower half seprately
	float t;
	if (((L1 - L0) * cusp.y - (cusp.x - L0) * C1) <= 0.f)
	{
		// Lower half

		t = cusp.y * L0 / (C1 * cusp.x + cusp.y * (L0 - L1));
	}
	else
	{
		// Upper half

		// First intersect with triangle
		t = cusp.y * (L0 - 1.f) / (C1 * (cusp.x - 1.f) + cusp.y * (L0 - L1));

		// Then one step Halley's method
		{
			float dL = L1 - L0;
			float dC = C1;

			float k_l = +0.3963377774f * a + 0.2158037573f * b;
			float k_m = -0.1055613458f * a - 0.0638541728f * b;
			float k_s = -0.0894841775f * a - 1.2914855480f * b;

			float l_dt = dL + dC * k_l;
			float m_dt = dL + dC * k_m;
			float s_dt = dL + dC * k_s;


			// If higher accuracy is required, 2 or 3 iterations of the following block can be used:
			{
				float L = L0 * (1.f - t) + t * L1;
				float C = t * C1;

				float l_ = L + C * k_l;
				float m_ = L + C * k_m;
				float s_ = L + C * k_s;

				float l = l_ * l_ * l_;
				float m = m_ * m_ * m_;
				float s = s_ * s_ * s_;

				float ldt = 3.f * l_dt * l_ * l_;
				float mdt = 3.f * m_dt * m_ * m_;
				float sdt = 3.f * s_dt * s_ * s_;

				float ldt2 = 6.f * l_dt * l_dt * l_;
				float mdt2 = 6.f * m_dt * m_dt * m_;
				float sdt2 = 6.f * s_dt * s_dt * s_;

				float r = 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s - 1.f;
				float r1 = 4.0767416621f * ldt - 3.3077115913f * mdt + 0.2309699292f * sdt;
				float r2 = 4.0767416621f * ldt2 - 3.3077115913f * mdt2 + 0.2309699292f * sdt2;

				float u_r = r1 / (r1 * r1 - 0.5f * r * r2);
				float t_r = -r * u_r;

				float g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s - 1.f;
				float g1 = -1.2684380046f * ldt + 2.6097574011f * mdt - 0.3413193965f * sdt;
				float g2 = -1.2684380046f * ldt2 + 2.6097574011f * mdt2 - 0.3413193965f * sdt2;

				float u_g = g1 / (g1 * g1 - 0.5f * g * g2);
				float t_g = -g * u_g;

				float b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s - 1.f;
				float b1 = -0.0041960863f * ldt - 0.7034186147f * mdt + 1.7076147010f * sdt;
				float b2 = -0.0041960863f * ldt2 - 0.7034186147f * mdt2 + 1.7076147010f * sdt2;

				float u_b = b1 / (b1 * b1 - 0.5f * b * b2);
				float t_b = -b * u_b;

				t_r = u_r >= 0.f ? t_r : 10000.f;
				t_g = u_g >= 0.f ? t_g : 10000.f;
				t_b = u_b >= 0.f ? t_b : 10000.f;

				t += min(t_r, min(t_g, t_b));
			}
		}
	}

	return t;
}

float find_gamut_intersection_5(float a, float b, float L1, float C1, float L0)
{
	// Find the cusp of the gamut triangle
	vec2 cusp = find_cusp(a, b);

	return find_gamut_intersection(a, b, L1, C1, L0, cusp);
}

vec3 gamut_clip_preserve_chroma(vec3 rgb)
{
	if (rgb.r < 1.f && rgb.g < 1.f && rgb.b < 1.f && rgb.r > 0.f && rgb.g > 0.f && rgb.b > 0.f)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001f;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L0 = clamp(L, 0.f, 1.f);

	float t = find_gamut_intersection_5(a_, b_, L, C, L0);
	float L_clipped = L0 * (1.f - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_project_to_0_5(vec3 rgb)
{
	if (rgb.r < 1.f && rgb.g < 1.f && rgb.b < 1.f && rgb.r > 0.f && rgb.g > 0.f && rgb.b > 0.f)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001f;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L0 = 0.5;

	float t = find_gamut_intersection_5(a_, b_, L, C, L0);
	float L_clipped = L0 * (1.f - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_project_to_L_cusp(vec3 rgb)
{
	if (rgb.r < 1.f && rgb.g < 1.f && rgb.b < 1.f && rgb.r > 0.f && rgb.g > 0.f && rgb.b > 0.f)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001f;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	// The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
	vec2 cusp = find_cusp(a_, b_);

	float L0 = cusp.x;

	float t = find_gamut_intersection_5(a_, b_, L, C, L0);

	float L_clipped = L0 * (1.f - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_adaptive_L0_0_5(vec3 rgb, float alpha)
{
	if (rgb.r < 1.f && rgb.g < 1.f && rgb.b < 1.f && rgb.r > 0.f && rgb.g > 0.f && rgb.b > 0.f)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001f;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float Ld = L - 0.5f;
	float e1 = 0.5f + abs(Ld) + alpha * C;
	float L0 = 0.5f * (1.f + sign(Ld) * (e1 - sqrt(e1 * e1 - 2.f * abs(Ld))));

	float t = find_gamut_intersection_5(a_, b_, L, C, L0);
	float L_clipped = L0 * (1.f - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

vec3 gamut_clip_adaptive_L0_L_cusp(vec3 rgb, float alpha)
{
	if (rgb.r < 1.f && rgb.g < 1.f && rgb.b < 1.f && rgb.r > 0.f && rgb.g > 0.f && rgb.b > 0.f)
		return rgb;

	vec3 lab = linear_srgb_to_oklab(rgb);

	float L = lab.x;
	float eps = 0.00001f;
	float C = max(eps, sqrt(lab.y * lab.y + lab.z * lab.z));
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	// The cusp is computed here and in find_gamut_intersection, an optimized solution would only compute it once.
	vec2 cusp = find_cusp(a_, b_);

	float Ld = L - cusp.x;
	float k = 2.f * (Ld > 0.f ? 1.f - cusp.x : cusp.x);

	float e1 = 0.5f * k + abs(Ld) + alpha * C / k;
	float L0 = cusp.x + 0.5f * (sign(Ld) * (e1 - sqrt(e1 * e1 - 2.f * k * abs(Ld))));

	float t = find_gamut_intersection_5(a_, b_, L, C, L0);
	float L_clipped = L0 * (1.f - t) + t * L;
	float C_clipped = t * C;

	return oklab_to_linear_srgb(vec3( L_clipped, C_clipped * a_, C_clipped * b_ ));
}

float toe(float x)
{
	float k_1 = 0.206f;
	float k_2 = 0.03f;
	float k_3 = (1.f + k_1) / (1.f + k_2);
	return 0.5f * (k_3 * x - k_1 + sqrt((k_3 * x - k_1) * (k_3 * x - k_1) + 4.f * k_2 * k_3 * x));
}

float toe_inv(float x)
{
	float k_1 = 0.206f;
	float k_2 = 0.03f;
	float k_3 = (1.f + k_1) / (1.f + k_2);
	return (x * x + k_1 * x) / (k_3 * (x + k_2));
}
 
vec2 to_ST(vec2 cusp)
{
	float L = cusp.x;
	float C = cusp.y;
	return vec2( C / L, C / (1.f - L) );
}

// Returns a smooth approximation of the location of the cusp
// This polynomial was created by an optimization process
// It has been designed so that S_mid < S_max and T_mid < T_max
vec2 get_ST_mid(float a_, float b_)
{
	float S = 0.11516993f + 1.f / (
		+7.44778970f + 4.15901240f * b_
		+ a_ * (-2.19557347f + 1.75198401f * b_
			+ a_ * (-2.13704948f - 10.02301043f * b_
				+ a_ * (-4.24894561f + 5.38770819f * b_ + 4.69891013f * a_
					)))
		);

	float T = 0.11239642f + 1.f / (
		+1.61320320f - 0.68124379f * b_
		+ a_ * (+0.40370612f + 0.90148123f * b_
			+ a_ * (-0.27087943f + 0.61223990f * b_
				+ a_ * (+0.00299215f - 0.45399568f * b_ - 0.14661872f * a_
					)))
		);

	return vec2( S, T );
}

vec3 get_Cs(float L, float a_, float b_)
{
	vec2 cusp = find_cusp(a_, b_);

	float C_max = find_gamut_intersection(a_, b_, L, 1.f, L, cusp);
	vec2 ST_max = to_ST(cusp);
	
	// Scale factor to compensate for the curved part of gamut shape:
	float k = C_max / min((L * ST_max.x), (1.f - L) * ST_max.y);

	float C_mid;
	{
		vec2 ST_mid = get_ST_mid(a_, b_);

		// Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
		float C_a = L * ST_mid.x;
		float C_b = (1.f - L) * ST_mid.y;
		C_mid = 0.9f * k * sqrt(sqrt(1.f / (1.f / (C_a * C_a * C_a * C_a) + 1.f / (C_b * C_b * C_b * C_b))));
	}

	float C_0;
	{
		// for C_0, the shape is independent of hue, so vec2 are constant. Values picked to roughly be the average values of vec2.
		float C_a = L * 0.4f;
		float C_b = (1.f - L) * 0.8f;

		// Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
		C_0 = sqrt(1.f / (1.f / (C_a * C_a) + 1.f / (C_b * C_b)));
	}

	return vec3( C_0, C_mid, C_max );
}

vec3 okhsl_to_srgb(vec3 hsl)
{
	float h = hsl.x;
	float s = hsl.y;
	float l = hsl.z;

	if (l == 1.0f)
	{
		return vec3( 1.f, 1.f, 1.f );
	}

	else if (l == 0.f)
	{
		return vec3( 0.f, 0.f, 0.f );
	}

	float a_ = cos(2.f * M_PI * h);
	float b_ = sin(2.f * M_PI * h);
	float L = toe_inv(l);

	vec3 cs = get_Cs(L, a_, b_);
	float C_0 = cs.x;
	float C_mid = cs.y;
	float C_max = cs.z;

	float mid = 0.8f;
	float mid_inv = 1.25f;

	float C, t, k_0, k_1, k_2;

	if (s < mid)
	{
		t = mid_inv * s;

		k_1 = mid * C_0;
		k_2 = (1.f - k_1 / C_mid);

		C = t * k_1 / (1.f - k_2 * t);
	}
	else
	{
		t = (s - mid)/ (1.f - mid);

		k_0 = C_mid;
		k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		k_2 = (1.f - (k_1) / (C_max - C_mid));

		C = k_0 + t * k_1 / (1.f - k_2 * t);
	}

	vec3 rgb = oklab_to_linear_srgb(vec3( L, C * a_, C * b_ ));
	return vec3(
		srgb_transfer_function(rgb.r),
		srgb_transfer_function(rgb.g),
		srgb_transfer_function(rgb.b)
	);
}

vec3 okhsl_to_linear_srgb(vec3 hsl)
{
	float h = hsl.x;
	float s = hsl.y;
	float l = hsl.z;

	if (l == 1.0f)
	{
		return vec3( 1.f, 1.f, 1.f );
	}

	else if (l == 0.f)
	{
		return vec3( 0.f, 0.f, 0.f );
	}

	float a_ = cos(2.f * M_PI * h);
	float b_ = sin(2.f * M_PI * h);
	float L = toe_inv(l);

	vec3 cs = get_Cs(L, a_, b_);
	float C_0 = cs.x;
	float C_mid = cs.y;
	float C_max = cs.z;

	float mid = 0.8f;
	float mid_inv = 1.25f;

	float C, t, k_0, k_1, k_2;

	if (s < mid)
	{
		t = mid_inv * s;

		k_1 = mid * C_0;
		k_2 = (1.f - k_1 / C_mid);

		C = t * k_1 / (1.f - k_2 * t);
	}
	else
	{
		t = (s - mid)/ (1.f - mid);

		k_0 = C_mid;
		k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		k_2 = (1.f - (k_1) / (C_max - C_mid));

		C = k_0 + t * k_1 / (1.f - k_2 * t);
	}

	vec3 rgb = oklab_to_linear_srgb(vec3( L, C * a_, C * b_ ));
	return rgb;
}

vec3 srgb_to_okhsl(vec3 rgb)
{
	vec3 lab = linear_srgb_to_oklab(vec3(
		srgb_transfer_function_inv(rgb.r),
		srgb_transfer_function_inv(rgb.g),
		srgb_transfer_function_inv(rgb.b)
		));

	float C = sqrt(lab.y * lab.y + lab.z * lab.z);
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L = lab.x;
	float h = 0.5f + 0.5f * atan(-lab.z, -lab.y) / M_PI;

	vec3 cs = get_Cs(L, a_, b_);
	float C_0 = cs.x;
	float C_mid = cs.y;
	float C_max = cs.z;

	// Inverse of the interpolation in okhsl_to_srgb:

	float mid = 0.8f;
	float mid_inv = 1.25f;

	float s;
	if (C < C_mid)
	{
		float k_1 = mid * C_0;
		float k_2 = (1.f - k_1 / C_mid);

		float t = C / (k_1 + k_2 * C);
		s = t * mid;
	}
	else
	{
		float k_0 = C_mid;
		float k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		float k_2 = (1.f - (k_1) / (C_max - C_mid));

		float t = (C - k_0) / (k_1 + k_2 * (C - k_0));
		s = mid + (1.f - mid) * t;
	}

	float l = toe(L);
	return vec3( h, s, l );
}

vec3 linear_srgb_to_okhsl(vec3 rgb)
{
	vec3 lab = linear_srgb_to_oklab(rgb);

	float C = sqrt(lab.y * lab.y + lab.z * lab.z);
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L = lab.x;
	float h = 0.5f + 0.5f * atan(-lab.z, -lab.y) / M_PI;

	vec3 cs = get_Cs(L, a_, b_);
	float C_0 = cs.x;
	float C_mid = cs.y;
	float C_max = cs.z;

	// Inverse of the interpolation in okhsl_to_srgb:

	float mid = 0.8f;
	float mid_inv = 1.25f;

	float s;
	if (C < C_mid)
	{
		float k_1 = mid * C_0;
		float k_2 = (1.f - k_1 / C_mid);

		float t = C / (k_1 + k_2 * C);
		s = t * mid;
	}
	else
	{
		float k_0 = C_mid;
		float k_1 = (1.f - mid) * C_mid * C_mid * mid_inv * mid_inv / C_0;
		float k_2 = (1.f - (k_1) / (C_max - C_mid));

		float t = (C - k_0) / (k_1 + k_2 * (C - k_0));
		s = mid + (1.f - mid) * t;
	}

	float l = toe(L);
	return vec3( h, s, l );
}


vec3 okhsv_to_srgb(vec3 hsv)
{
	float h = hsv.x;
	float s = hsv.y;
	float v = hsv.z;

	float a_ = cos(2.f * M_PI * h);
	float b_ = sin(2.f * M_PI * h);
	
	vec2 cusp = find_cusp(a_, b_);
	vec2 ST_max = to_ST(cusp);
	float S_max = ST_max.x;
	float T_max = ST_max.y;
	float S_0 = 0.5f;
	float k = 1.f- S_0 / S_max;

	// first we compute L and V as if the gamut is a perfect triangle:

	// L, C when v==1:
	float L_v = 1.f   - s * S_0 / (S_0 + T_max - T_max * k * s);
	float C_v = s * T_max * S_0 / (S_0 + T_max - T_max * k * s);

	float L = v * L_v;
	float C = v * C_v;

	// then we compensate for both toe and the curved top part of the triangle:
	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;

	float L_new = toe_inv(L);
	C = C * L_new / L;
	L = L_new;

	vec3 rgb_scale = oklab_to_linear_srgb(vec3( L_vt, a_ * C_vt, b_ * C_vt ));
	float scale_L = cbrt(1.f / max(max(rgb_scale.r, rgb_scale.g), max(rgb_scale.b, 0.f)));

	L = L * scale_L;
	C = C * scale_L;

	vec3 rgb = oklab_to_linear_srgb(vec3( L, C * a_, C * b_ ));
	return vec3(
		srgb_transfer_function(rgb.r),
		srgb_transfer_function(rgb.g),
		srgb_transfer_function(rgb.b)
	);
}

vec3 srgb_to_okhsv(vec3 rgb)
{
	vec3 lab = linear_srgb_to_oklab(vec3(
		srgb_transfer_function_inv(rgb.r),
		srgb_transfer_function_inv(rgb.g),
		srgb_transfer_function_inv(rgb.b)
		));

	float C = sqrt(lab.y * lab.y + lab.z * lab.z);
	float a_ = lab.y / C;
	float b_ = lab.z / C;

	float L = lab.x;
	float h = 0.5f + 0.5f * atan(-lab.z, -lab.y) / M_PI;

	vec2 cusp = find_cusp(a_, b_);
	vec2 ST_max = to_ST(cusp);
	float S_max = ST_max.x;
	float T_max = ST_max.y;
	float S_0 = 0.5f;
	float k = 1.f - S_0 / S_max;

	// first we find L_v, C_v, L_vt and C_vt

	float t = T_max / (C + L * T_max);
	float L_v = t * L;
	float C_v = t * C;

	float L_vt = toe_inv(L_v);
	float C_vt = C_v * L_vt / L_v;

	// we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
	vec3 rgb_scale = oklab_to_linear_srgb(vec3( L_vt, a_ * C_vt, b_ * C_vt ));
	float scale_L = cbrt(1.f / max(max(rgb_scale.r, rgb_scale.g), max(rgb_scale.b, 0.f)));

	L = L / scale_L;
	C = C / scale_L;

	C = C * toe(L) / L;
	L = toe(L);

	// we can now compute v and s:

	float v = L / L_v;
	float s = (S_0 + T_max) * C_v / ((T_max * S_0) + T_max * k * C_v);

	return vec3 (h, s, v );
}












vec3 tonemap_filmic(vec3 color, float white) {
	// exposure bias: input scale (color *= bias, white *= bias) to make the brightness consistent with other tonemappers
	// also useful to scale the input to the range that the tonemapper is designed for (some require very high input values)
	// has no effect on the curve's general shape or visual properties
	const float exposure_bias = 2.0f;
	const float A = 0.22f * exposure_bias * exposure_bias; // bias baked into constants for performance
	const float B = 0.30f * exposure_bias;
	const float C = 0.10f;
	const float D = 0.20f;
	const float E = 0.01f;
	const float F = 0.30f;

	vec3 color_tonemapped = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - E / F;
	float white_tonemapped = ((white * (A * white + C * B) + D * E) / (white * (A * white + B) + D * F)) - E / F;

	return color_tonemapped / white_tonemapped;
}

// Adapted from https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
// (MIT License).
vec3 tonemap_aces(vec3 color, float white) {
	const float exposure_bias = 1.8f;
	const float A = 0.0245786f;
	const float B = 0.000090537f;
	const float C = 0.983729f;
	const float D = 0.432951f;
	const float E = 0.238081f;

	// Exposure bias baked into transform to save shader instructions. Equivalent to `color *= exposure_bias`
	const mat3 rgb_to_rrt = mat3(
			vec3(0.59719f * exposure_bias, 0.35458f * exposure_bias, 0.04823f * exposure_bias),
			vec3(0.07600f * exposure_bias, 0.90834f * exposure_bias, 0.01566f * exposure_bias),
			vec3(0.02840f * exposure_bias, 0.13383f * exposure_bias, 0.83777f * exposure_bias));

	const mat3 odt_to_rgb = mat3(
			vec3(1.60475f, -0.53108f, -0.07367f),
			vec3(-0.10208f, 1.10813f, -0.00605f),
			vec3(-0.00327f, -0.07276f, 1.07602f));

	color *= rgb_to_rrt;
	vec3 color_tonemapped = (color * (color + A) - B) / (color * (C * color + D) + E);
	color_tonemapped *= odt_to_rgb;

	white *= exposure_bias;
	float white_tonemapped = (white * (white + A) - B) / (white * (C * white + D) + E);

	return color_tonemapped / white_tonemapped;
}

vec3 convertRGB2XYZ(vec3 _rgb)
{
	// Reference:
	// RGB/XYZ Matrices
	// http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
	vec3 xyz;
	xyz.x = dot(vec3(0.4124564, 0.3575761, 0.1804375), _rgb);
	xyz.y = dot(vec3(0.2126729, 0.7151522, 0.0721750), _rgb);
	xyz.z = dot(vec3(0.0193339, 0.1191920, 0.9503041), _rgb);
	return xyz;
}

vec3 convertXYZ2RGB(vec3 _xyz)
{
	vec3 rgb;
	rgb.x = dot(vec3( 3.2404542, -1.5371385, -0.4985314), _xyz);
	rgb.y = dot(vec3(-0.9692660,  1.8760108,  0.0415560), _xyz);
	rgb.z = dot(vec3( 0.0556434, -0.2040259,  1.0572252), _xyz);
	return rgb;
}

vec3 tonemap_reinhard(vec3 color, float white) {
	float srcLum = convertRGB2XYZ(color).y;
	if (srcLum > 0.0)
	{
		float dstLum = (white * srcLum + srcLum) / (srcLum * white + white);
		color = color * (dstLum / srcLum);
	}

	float distance = 0.0;
	
	// Clamp from 0 to 1 without changing hue of color
	// Only workes with "Mix" or "Replace" glow modes because other modes are applied to the
	// base scene after it has already been tone mapped.
	// Doesn't work as well on mobile because the max value of a color component is limited
	if (color.r > 1.0 || color.g > 1.0 || color.b > 1.0)
	{
		if (color.r > color.g && color.r > color.b)
		{
			distance = color.r - 1.0;
			color = color / color.r;
		}
		else if (color.g > color.r && color.g > color.b)
		{
			distance = color.g - 1.0;
			color = color / color.g;
		}
		else
		{
			distance = color.b - 1.0;
			color = color / color.b;
		}
	}
	
	vec3 hsl = linear_srgb_to_okhsl(color);
	hsl.z += distance / 10.0; // divided by an arbitrary number for this rough prototype
	hsl.z = min(hsl.z, 1.0);
	color = okhsl_to_linear_srgb(hsl);

	return color;
}

vec3 linear_to_srgb(vec3 color) {
	//if going to srgb, clamp from 0 to 1.
	color = clamp(color, vec3(0.0), vec3(1.0));
	const vec3 a = vec3(0.055f);
	return mix((vec3(1.0f) + a) * pow(color.rgb, vec3(1.0f / 2.4f)) - a, 12.92f * color.rgb, lessThan(color.rgb, vec3(0.0031308f)));
}

#define TONEMAPPER_LINEAR 0
#define TONEMAPPER_REINHARD 1
#define TONEMAPPER_FILMIC 2
#define TONEMAPPER_ACES 3

vec3 apply_tonemapping(vec3 color, float white) { // inputs are LINEAR, always outputs clamped [0;1] color
	// Ensure color values passed to tonemappers are positive.
	// They can be negative in the case of negative lights, which leads to undesired behavior.
	if (params.tonemapper == TONEMAPPER_LINEAR) {
		
		float distance = 0.0;

		// Clamp from 0 to 1 without changing hue of color
		// Only workes with "Mix" or "Replace" glow modes because other modes are applied to the
		// base scene after it has already been tone mapped.
		// Doesn't work as well on mobile because the max value of a color component is limited
		if (color.r > 1.0 || color.g > 1.0 || color.b > 1.0)
		{
			if (color.r > color.g && color.r > color.b)
			{
				distance = color.r - 1.0;
				color = color / color.r;
			}
			else if (color.g > color.r && color.g > color.b)
			{
				distance = color.g - 1.0;
				color = color / color.g;
			}
			else
			{
				distance = color.b - 1.0;
				color = color / color.b;
			}
		}

		vec3 hsl = linear_srgb_to_okhsl(color);
		hsl.z += distance / 10.0; // divided by an arbitrary number for this rough prototype
		hsl.z = min(hsl.z, 1.0);
		color = okhsl_to_linear_srgb(hsl);

		return color;
	} else if (params.tonemapper == TONEMAPPER_REINHARD) {
		return tonemap_reinhard(max(vec3(0.0f), color), white);
	} else if (params.tonemapper == TONEMAPPER_FILMIC) {
		return tonemap_filmic(max(vec3(0.0f), color), white);
	} else { // TONEMAPPER_ACES
		return tonemap_aces(max(vec3(0.0f), color), white);
	}
}

#ifdef MULTIVIEW
vec3 gather_glow(sampler2DArray tex, vec2 uv) { // sample all selected glow levels, view is added to uv later
#else
vec3 gather_glow(sampler2D tex, vec2 uv) { // sample all selected glow levels
#endif // defined(MULTIVIEW)
	vec3 glow = vec3(0.0f);

	if (params.glow_levels[0] > 0.0001) {
		glow += GLOW_TEXTURE_SAMPLE(tex, uv, 0).rgb * params.glow_levels[0];
	}

	if (params.glow_levels[1] > 0.0001) {
		glow += GLOW_TEXTURE_SAMPLE(tex, uv, 1).rgb * params.glow_levels[1];
	}

	if (params.glow_levels[2] > 0.0001) {
		glow += GLOW_TEXTURE_SAMPLE(tex, uv, 2).rgb * params.glow_levels[2];
	}

	if (params.glow_levels[3] > 0.0001) {
		glow += GLOW_TEXTURE_SAMPLE(tex, uv, 3).rgb * params.glow_levels[3];
	}

	if (params.glow_levels[4] > 0.0001) {
		glow += GLOW_TEXTURE_SAMPLE(tex, uv, 4).rgb * params.glow_levels[4];
	}

	if (params.glow_levels[5] > 0.0001) {
		glow += GLOW_TEXTURE_SAMPLE(tex, uv, 5).rgb * params.glow_levels[5];
	}

	if (params.glow_levels[6] > 0.0001) {
		glow += GLOW_TEXTURE_SAMPLE(tex, uv, 6).rgb * params.glow_levels[6];
	}

	return glow;
}

#define GLOW_MODE_ADD 0
#define GLOW_MODE_SCREEN 1
#define GLOW_MODE_SOFTLIGHT 2
#define GLOW_MODE_REPLACE 3
#define GLOW_MODE_MIX 4

vec3 apply_glow(vec3 color, vec3 glow) { // apply glow using the selected blending mode
	if (params.glow_mode == GLOW_MODE_ADD) {
		return color + glow;
	} else if (params.glow_mode == GLOW_MODE_SCREEN) {
		//need color clamping
		return max((color + glow) - (color * glow), vec3(0.0));
	} else if (params.glow_mode == GLOW_MODE_SOFTLIGHT) {
		//need color clamping
		glow = glow * vec3(0.5f) + vec3(0.5f);

		color.r = (glow.r <= 0.5f) ? (color.r - (1.0f - 2.0f * glow.r) * color.r * (1.0f - color.r)) : (((glow.r > 0.5f) && (color.r <= 0.25f)) ? (color.r + (2.0f * glow.r - 1.0f) * (4.0f * color.r * (4.0f * color.r + 1.0f) * (color.r - 1.0f) + 7.0f * color.r)) : (color.r + (2.0f * glow.r - 1.0f) * (sqrt(color.r) - color.r)));
		color.g = (glow.g <= 0.5f) ? (color.g - (1.0f - 2.0f * glow.g) * color.g * (1.0f - color.g)) : (((glow.g > 0.5f) && (color.g <= 0.25f)) ? (color.g + (2.0f * glow.g - 1.0f) * (4.0f * color.g * (4.0f * color.g + 1.0f) * (color.g - 1.0f) + 7.0f * color.g)) : (color.g + (2.0f * glow.g - 1.0f) * (sqrt(color.g) - color.g)));
		color.b = (glow.b <= 0.5f) ? (color.b - (1.0f - 2.0f * glow.b) * color.b * (1.0f - color.b)) : (((glow.b > 0.5f) && (color.b <= 0.25f)) ? (color.b + (2.0f * glow.b - 1.0f) * (4.0f * color.b * (4.0f * color.b + 1.0f) * (color.b - 1.0f) + 7.0f * color.b)) : (color.b + (2.0f * glow.b - 1.0f) * (sqrt(color.b) - color.b)));
		return color;
	} else { //replace
		return glow;
	}
}

vec3 apply_bcs(vec3 color, vec3 bcs) {
	color = mix(vec3(0.0f), color, bcs.x);
	color = mix(vec3(0.5f), color, bcs.y);
	color = mix(vec3(dot(vec3(1.0f), color) * 0.33333f), color, bcs.z);

	return color;
}
#ifdef USE_1D_LUT
vec3 apply_color_correction(vec3 color) {
	color.r = texture(source_color_correction, vec2(color.r, 0.0f)).r;
	color.g = texture(source_color_correction, vec2(color.g, 0.0f)).g;
	color.b = texture(source_color_correction, vec2(color.b, 0.0f)).b;
	return color;
}
#else
vec3 apply_color_correction(vec3 color) {
	return textureLod(source_color_correction, color, 0.0).rgb;
}
#endif

#ifndef SUBPASS
vec3 do_fxaa(vec3 color, float exposure, vec2 uv_interp) {
	const float FXAA_REDUCE_MIN = (1.0 / 128.0);
	const float FXAA_REDUCE_MUL = (1.0 / 8.0);
	const float FXAA_SPAN_MAX = 8.0;

#ifdef MULTIVIEW
	vec3 rgbNW = textureLod(source_color, vec3(uv_interp + vec2(-0.5, -0.5) * params.pixel_size, ViewIndex), 0.0).xyz * exposure * params.luminance_multiplier;
	vec3 rgbNE = textureLod(source_color, vec3(uv_interp + vec2(0.5, -0.5) * params.pixel_size, ViewIndex), 0.0).xyz * exposure * params.luminance_multiplier;
	vec3 rgbSW = textureLod(source_color, vec3(uv_interp + vec2(-0.5, 0.5) * params.pixel_size, ViewIndex), 0.0).xyz * exposure * params.luminance_multiplier;
	vec3 rgbSE = textureLod(source_color, vec3(uv_interp + vec2(0.5, 0.5) * params.pixel_size, ViewIndex), 0.0).xyz * exposure * params.luminance_multiplier;
#else
	vec3 rgbNW = textureLod(source_color, uv_interp + vec2(-0.5, -0.5) * params.pixel_size, 0.0).xyz * exposure * params.luminance_multiplier;
	vec3 rgbNE = textureLod(source_color, uv_interp + vec2(0.5, -0.5) * params.pixel_size, 0.0).xyz * exposure * params.luminance_multiplier;
	vec3 rgbSW = textureLod(source_color, uv_interp + vec2(-0.5, 0.5) * params.pixel_size, 0.0).xyz * exposure * params.luminance_multiplier;
	vec3 rgbSE = textureLod(source_color, uv_interp + vec2(0.5, 0.5) * params.pixel_size, 0.0).xyz * exposure * params.luminance_multiplier;
#endif
	vec3 rgbM = color;
	vec3 luma = vec3(0.299, 0.587, 0.114);
	float lumaNW = dot(rgbNW, luma);
	float lumaNE = dot(rgbNE, luma);
	float lumaSW = dot(rgbSW, luma);
	float lumaSE = dot(rgbSE, luma);
	float lumaM = dot(rgbM, luma);
	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

	vec2 dir;
	dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
	dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

	float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) *
					(0.25 * FXAA_REDUCE_MUL),
			FXAA_REDUCE_MIN);

	float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
	dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX),
				  max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX),
						  dir * rcpDirMin)) *
			params.pixel_size;

#ifdef MULTIVIEW
	vec3 rgbA = 0.5 * exposure * (textureLod(source_color, vec3(uv_interp + dir * (1.0 / 3.0 - 0.5), ViewIndex), 0.0).xyz + textureLod(source_color, vec3(uv_interp + dir * (2.0 / 3.0 - 0.5), ViewIndex), 0.0).xyz) * params.luminance_multiplier;
	vec3 rgbB = rgbA * 0.5 + 0.25 * exposure * (textureLod(source_color, vec3(uv_interp + dir * -0.5, ViewIndex), 0.0).xyz + textureLod(source_color, vec3(uv_interp + dir * 0.5, ViewIndex), 0.0).xyz) * params.luminance_multiplier;
#else
	vec3 rgbA = 0.5 * exposure * (textureLod(source_color, uv_interp + dir * (1.0 / 3.0 - 0.5), 0.0).xyz + textureLod(source_color, uv_interp + dir * (2.0 / 3.0 - 0.5), 0.0).xyz) * params.luminance_multiplier;
	vec3 rgbB = rgbA * 0.5 + 0.25 * exposure * (textureLod(source_color, uv_interp + dir * -0.5, 0.0).xyz + textureLod(source_color, uv_interp + dir * 0.5, 0.0).xyz) * params.luminance_multiplier;
#endif

	float lumaB = dot(rgbB, luma);
	if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
		return rgbA;
	} else {
		return rgbB;
	}
}
#endif // !SUBPASS

// From https://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf
// and https://www.shadertoy.com/view/MslGR8 (5th one starting from the bottom)
// NOTE: `frag_coord` is in pixels (i.e. not normalized UV).
vec3 screen_space_dither(vec2 frag_coord) {
	// Iestyn's RGB dither (7 asm instructions) from Portal 2 X360, slightly modified for VR.
	vec3 dither = vec3(dot(vec2(171.0, 231.0), frag_coord));
	dither.rgb = fract(dither.rgb / vec3(103.0, 71.0, 97.0));

	// Subtract 0.5 to avoid slightly brightening the whole viewport.
	return (dither.rgb - 0.5) / 255.0;
}

void main() {
#ifdef SUBPASS
	// SUBPASS and MULTIVIEW can be combined but in that case we're already reading from the correct layer
	vec4 color = subpassLoad(input_color);
#elif defined(MULTIVIEW)
	vec4 color = textureLod(source_color, vec3(uv_interp, ViewIndex), 0.0f);
#else
	vec4 color = textureLod(source_color, uv_interp, 0.0f);
#endif
	color.rgb *= params.luminance_multiplier;

	// Exposure

	float exposure = params.exposure;

#ifndef SUBPASS
	if (params.use_auto_exposure) {
		exposure *= 1.0 / (texelFetch(source_auto_exposure, ivec2(0, 0), 0).r * params.luminance_multiplier / params.auto_exposure_scale);
	}
#endif

	color.rgb *= exposure;

	// Early Tonemap & SRGB Conversion
#ifndef SUBPASS
	if (params.use_fxaa) {
		// FXAA must be performed before glow to preserve the "bleed" effect of glow.
		color.rgb = do_fxaa(color.rgb, exposure, uv_interp);
	}

	if (params.use_glow && params.glow_mode == GLOW_MODE_MIX) {
		vec3 glow = gather_glow(source_glow, uv_interp) * params.luminance_multiplier;
		if (params.glow_map_strength > 0.001) {
			glow = mix(glow, texture(glow_map, uv_interp).rgb * glow, params.glow_map_strength);
		}
		color.rgb = mix(color.rgb, glow, params.glow_intensity);
	}
#endif

	color.rgb = apply_tonemapping(color.rgb, params.white);

	color.rgb = linear_to_srgb(color.rgb); // regular linear -> SRGB conversion

#ifndef SUBPASS
	// Glow
	if (params.use_glow && params.glow_mode != GLOW_MODE_MIX) {
		vec3 glow = gather_glow(source_glow, uv_interp) * params.glow_intensity * params.luminance_multiplier;
		if (params.glow_map_strength > 0.001) {
			glow = mix(glow, texture(glow_map, uv_interp).rgb * glow, params.glow_map_strength);
		}

		// high dynamic range -> SRGB
		glow = apply_tonemapping(glow, params.white);
		glow = linear_to_srgb(glow);

		color.rgb = apply_glow(color.rgb, glow);
	}
#endif

	// Additional effects

	if (params.use_bcs) {
		color.rgb = apply_bcs(color.rgb, params.bcs);
	}

	if (params.use_color_correction) {
		color.rgb = apply_color_correction(color.rgb);
	}

	if (params.use_debanding) {
		// Debanding should be done at the end of tonemapping, but before writing to the LDR buffer.
		// Otherwise, we're adding noise to an already-quantized image.
		color.rgb += screen_space_dither(gl_FragCoord.xy);
	}

	frag_color = color;
}
