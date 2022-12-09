
#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

struct ObjDesc
{
    int txtOffset;
    uint64_t vertexAddress;
    uint64_t indexAddress;
    uint64_t materialAddress;
    uint64_t materialIndexAddress;
};

struct GlobalUniforms
{
    mat4 viewProj;
    mat4 viewInverse;
    mat4 projInverse;
};

struct PushConstantRaster
{
    mat4 modelMatrix   ;
    vec3 lightPosition ;
    uint objIndex      ;
    float lightIntensity;
    int lightType     ;
};

struct PushConstantRay
{
    vec4 clearColor;
    vec3 lightPosition;
    float lightIntensity;
    int lightType;
    int maxDepth;
};

struct Vertex
{
    vec3 pos;
    vec3 nrm;
    vec3 color;
    vec2 texCoord;
};

struct WaveFrontMaterial
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 transmittance;
    vec3 emission;
    float shininess;
    float ior;
    float dissolve;
    int illum;
    int textureId;
};

struct hitPayload
{
    vec3 hitValue;
    int depth;
    vec3 attenuation;
    int done;
    vec3 rayOrigin;
    vec3 rayDir;
};

layout (location = 0) out vec4 outColor;
layout (location = 3) in vec3 fragColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
