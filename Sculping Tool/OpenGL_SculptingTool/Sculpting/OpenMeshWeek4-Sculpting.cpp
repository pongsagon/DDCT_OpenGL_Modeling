#include <GLFW/glfw3.h>
#include <glad/glad.h>


#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <queue>
#include <unordered_set>
#include <chrono> 

#include <unordered_map>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Geometry/VectorT.hh>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "camera.h"
#include "shader.h"

#include <algorithm>  
#define NOMINMAX     
#include <windows.h>

#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>



using namespace std::chrono;
typedef OpenMesh::TriMesh_ArrayKernelT<> MyMesh;

const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

MyMesh mesh;
Camera camera(glm::vec3(0.0f, 0.0f, 5.0f));  // Start close to the sphere
Shader* shaderProgram;
Shader* pickingShader;
GLuint circleVAO = 0, circleVBO = 0;
GLuint pointVAO = 0, pointVBO = 0;
Shader* debugShader;  // Simple shader to draw color-only lines/points
float lastX = SCR_WIDTH / 2.0f, lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f, lastFrame = 0.0f;

GLuint VAO, VBO, EBO;
std::vector<float> vertices;
std::vector<unsigned int> indices;
bool isLooking = false;  // ✅ Track middle mouse press


float brushSize = 0.2f;  // Initial brush size
bool isSculptingUp = false;
bool isSculptingDown = false;
bool meshDirty = false;

glm::vec3 sculptIndicatorPosition;
glm::vec3 sculptIndicatorNormal;
bool sculptIndicatorVisible = false;
std::vector<glm::vec3> circleVertices;
std::vector<bool> affectedTriangleFlags;
//std::vector<size_t> affectedTriangleIndices;
bool wireframeMode = false;  // Global toggle flag
MyMesh::VertexHandle sculptIndicatorVertex; // Declare globally if appropriate
std::vector<MyMesh::VertexHandle> vertexHandleMapping; // Global mapping
std::unordered_map<int, std::vector<size_t>> vertexToTriangleMap;
int frameCount = 0;
double previousTime = 0.0;

bool isOrbiting = false;
float yaw = -90.0f, pitch = 0.0f, distance = 5.0f;
glm::vec3 orbitCenter(0.0f, 0.0f, 0.0f);

float brushStrength = 0.005f;  // Default sculpt strength
const float MIN_BRUSH_STRENGTH = 0.001f;
const float MAX_BRUSH_STRENGTH = 0.1f;


bool snakeHookMode = false;       // 🔁 Toggle ON/OFF with 'S'
bool isSnakeHookDragging = false; // 🖱️ True only during drag

glm::vec2 lastDragPos;
glm::vec2 currentDragPos;
bool lockSculptIndicator = false;
glm::vec3 snakeHookStartProjected;

// Already partially declared:
std::unordered_set<size_t> affectedTriangles; // used in sculptMesh
std::unordered_map<size_t, glm::vec3> triangleNormals; // store face normals
std::vector<glm::vec3> faceCenterCache;     // cache for face centers
std::vector<glm::vec3> faceNormalCache;     // cache for face normals
std::vector<float> faceAreaCache;           // cache for triangle areas
bool faceCacheValid = false;                // flag to avoid recomputing each frame
std::vector<glm::vec3> allFaceCenters;
std::vector<glm::vec3> allFaceNormals;
std::vector<float> allFaceAreas;

int currentSubdivisionLevel = 3;
const int maxSubdivision = 8;

bool isPulling = false;
bool pullActive = false;
glm::vec2 pullStartScreenPos;
glm::vec3 pullStartWorldPos;
MyMesh::VertexHandle pullVertex;



float gaussian(float distance, float sigma = 0.4f) {
    float exponent = -pow(distance, 2) / (2.0f * pow(sigma, 2));
    return exp(exponent);
}


void generateIcosphere(int subdivisions = 2, float radius = 1.0f) {
    mesh.clear();
    vertices.clear();
    indices.clear();
    mesh.request_vertex_normals();
    // Golden ratio for initial icosahedron
    const float t = (1.0f + sqrt(5.0f)) / 2.0f;

    std::vector<glm::vec3> baseVertices = {
        {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
        {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
        {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}
    };

    // Normalize to unit sphere
    for (auto& v : baseVertices)
        v = glm::normalize(v) * radius;

    // Create initial icosahedron faces
    std::vector<std::array<int, 3>> baseFaces = {
        {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
        {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
        {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
        {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
    };

    std::unordered_map<uint64_t, int> midpointCache;

    auto getMidpoint = [&](int v1, int v2) -> int {
        uint64_t key = ((uint64_t)std::min(v1, v2) << 32) | std::max(v1, v2);
        if (midpointCache.count(key)) return midpointCache[key];

        glm::vec3 mid = glm::normalize((baseVertices[v1] + baseVertices[v2]) * 0.5f) * radius;
        int newIndex = baseVertices.size();
        baseVertices.push_back(mid);
        midpointCache[key] = newIndex;
        return newIndex;
        };

    for (int s = 0; s < subdivisions; s++) {
        std::cout << "Subdividing Icosphere... " << s + 1 << "/" << subdivisions << "\n";
        std::vector<std::array<int, 3>> newFaces;
        for (auto& f : baseFaces) {
            int a = getMidpoint(f[0], f[1]);
            int b = getMidpoint(f[1], f[2]);
            int c = getMidpoint(f[2], f[0]);

            newFaces.push_back({ f[0], a, c });
            newFaces.push_back({ f[1], b, a });
            newFaces.push_back({ f[2], c, b });
            newFaces.push_back({ a, b, c });
        }
        baseFaces = std::move(newFaces);
    }

    std::vector<MyMesh::VertexHandle> vertexHandles;
    for (auto& v : baseVertices) {
        vertexHandles.push_back(mesh.add_vertex(MyMesh::Point(v.x, v.y, v.z)));
    }

    for (auto& f : baseFaces) {
        mesh.add_face(vertexHandles[f[0]], vertexHandles[f[1]], vertexHandles[f[2]]);
    }

    mesh.update_normals();

    std::cout << "Generated Icosphere with " << mesh.n_vertices() << " vertices and " << mesh.n_faces() << " faces.\n";
}




glm::vec3 toVec3(const MyMesh::Point& p) {
    return glm::vec3(p[0], p[1], p[2]);
}


void precomputeAllFaceData() {
    allFaceCenters.clear();
    allFaceNormals.clear();
    allFaceAreas.clear();

    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        glm::vec3 triCenter(0.0f);
        std::vector<glm::vec3> verts;

        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
            glm::vec3 p = toVec3(mesh.point(*fv_it));
            triCenter += p;
            verts.push_back(p);
        }

        triCenter /= 3.0f;
        glm::vec3 normal = glm::normalize(glm::cross(verts[1] - verts[0], verts[2] - verts[0]));
        float area = 0.5f * glm::length(glm::cross(verts[1] - verts[0], verts[2] - verts[0]));

        allFaceCenters.push_back(triCenter);
        allFaceNormals.push_back(normal);
        allFaceAreas.push_back(area);
    }

    std::cout << "[Init] Cached " << allFaceCenters.size() << " triangle face attributes.\n";
}


// Call this ONLY once during initialization
void setupMeshBuffers() {
    vertices.clear();
    indices.clear();
    vertexHandleMapping.clear();
    
    unsigned int indexCounter = 0;

    for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
        std::vector<MyMesh::VertexHandle> faceVertexHandles;
        for (auto fv_it = mesh.fv_iter(*f_it); fv_it.is_valid(); ++fv_it)
            faceVertexHandles.push_back(*fv_it);

        glm::vec3 v0 = toVec3(mesh.point(faceVertexHandles[0]));
        glm::vec3 v1 = toVec3(mesh.point(faceVertexHandles[1]));
        glm::vec3 v2 = toVec3(mesh.point(faceVertexHandles[2]));
        glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        for (int i = 0; i < 3; ++i) {
            glm::vec3 pos = toVec3(mesh.point(faceVertexHandles[i]));
            vertices.insert(vertices.end(), {
                pos.x, pos.y, pos.z,
                faceNormal.x, faceNormal.y, faceNormal.z
                });
            indices.push_back(indexCounter);
            vertexHandleMapping.push_back(faceVertexHandles[i]);

            // Store mapping from vertex to triangle index
            vertexToTriangleMap[faceVertexHandles[i].idx()].push_back(indexCounter / 3); // triangle index
            indexCounter++;
        }
    }

    affectedTriangleFlags.resize(indices.size() / 3, false);

    std::vector<float> pickingColors;
    for (size_t i = 0; i < vertexHandleMapping.size(); i += 3) {
        int triangleID = i / 3;
        float r = ((triangleID >> 0) & 0xFF) / 255.0f;
        float g = ((triangleID >> 8) & 0xFF) / 255.0f;
        float b = ((triangleID >> 16) & 0xFF) / 255.0f;

        for (int j = 0; j < 3; ++j) {
            pickingColors.push_back(r);
            pickingColors.push_back(g);
            pickingColors.push_back(b);
        }
    }


    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    GLuint colorVBO;
    glGenBuffers(1, &colorVBO);  // ✅ Move this before the VAO block too

    glBindVertexArray(VAO);

    // Vertex positions & normals
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Triangle ID (picking color)
    glBindBuffer(GL_ARRAY_BUFFER, colorVBO);
    glBufferData(GL_ARRAY_BUFFER, pickingColors.size() * sizeof(float), pickingColors.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(2);

    // Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_DYNAMIC_DRAW);

    glBindVertexArray(0);

}

void subdivideMesh(int level) {
    if (level < 0) return;

    MyMesh originalMesh = mesh; // Backup

    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_vertex_status();

    OpenMesh::Subdivider::Uniform::LoopT<MyMesh> loop;
    loop.attach(mesh);

    std::cout << "Applying " << level << " Loop subdivisions..." << std::endl;
    loop(level);
    loop.detach();

    mesh.garbage_collection();
    mesh.update_normals();

    setupMeshBuffers();
    precomputeAllFaceData();

    std::cout << "Subdivision complete. New vertex count: " << mesh.n_vertices() << std::endl;
}

void filterLocalFaceCache(const glm::vec3& center, float radius) {
    faceCenterCache.clear();
    faceNormalCache.clear();
    faceAreaCache.clear();

    float r2 = radius * radius;

    for (size_t i = 0; i < allFaceCenters.size(); ++i) {
        if (glm::distance2(allFaceCenters[i], center) <= r2) {
            faceCenterCache.push_back(allFaceCenters[i]);
            faceNormalCache.push_back(allFaceNormals[i]);
            faceAreaCache.push_back(allFaceAreas[i]);
        }
    }

   
}




void setupIndicatorframebuffer()
{
    glGenVertexArrays(1, &pointVAO);
    glGenBuffers(1, &pointVBO);
    glBindVertexArray(pointVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
    glEnableVertexAttribArray(0);

}



float getScaledBrushSize() {
    // Compute distance between camera and orbit center (or sculpt point if visible)
    glm::vec3 targetPoint = sculptIndicatorVisible ? sculptIndicatorPosition : orbitCenter;
    float dist = glm::length(camera.Position - targetPoint);

    // Scale factor — tweak multiplier if needed to match feeling
    float scaleFactor = 0.2f;
    return brushSize * dist * scaleFactor;
}


void drawCircleIndicator(glm::vec3 position, glm::vec3 normal, float radius, glm::vec3 color) {
    const int numSegments = 64;
    if (circleVAO == 0) {
        glGenVertexArrays(1, &circleVAO);
        glGenBuffers(1, &circleVBO);

        glBindVertexArray(circleVAO);
        glBindBuffer(GL_ARRAY_BUFFER, circleVBO);
        glBufferData(GL_ARRAY_BUFFER, (numSegments + 1) * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);
    }

    circleVertices.clear();

    // Create tangent and bitangent orthogonal to the normal
   // Ensure orthonormal basis around the normal (up = normal)
    glm::vec3 up = glm::normalize(normal);

    // Choose arbitrary vector not parallel to normal
    glm::vec3 temp = (fabs(up.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);

    // Gram-Schmidt process
    glm::vec3 tangent = glm::normalize(temp - up * glm::dot(temp, up));
    glm::vec3 bitangent = glm::normalize(glm::cross(up, tangent));

    for (int i = 0; i <= numSegments; ++i) {
        float theta = (float)i / numSegments * 2.0f * M_PI;
        glm::vec3 point = position + radius * (cos(theta) * tangent + sin(theta) * bitangent);
        circleVertices.push_back(point);
    }

    glBindBuffer(GL_ARRAY_BUFFER, circleVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, circleVertices.size() * sizeof(glm::vec3), circleVertices.data());

    debugShader->use();
    glm::mat4 MVP = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 1000.0f)
        * camera.GetViewMatrix()
        * glm::mat4(1.0f);
    debugShader->setMat4("MVP", MVP);
    debugShader->setVec3("color", color);

    glBindVertexArray(circleVAO);
    glDrawArrays(GL_LINE_STRIP, 0, circleVertices.size());



}

glm::vec3 updatedIndicatorNormal; // New global or static variable to store computed normal

void drawSculptIndicatorAndBrushArea() {
    if (!sculptIndicatorVisible) return;

    float scaledBrushSize = getScaledBrushSize();
    glm::vec3 offset = updatedIndicatorNormal * getScaledBrushSize() * 0.05f;
    glm::vec3 adjustedPosition = sculptIndicatorPosition + offset;

    // Draw brush circles
    drawCircleIndicator(adjustedPosition, updatedIndicatorNormal, scaledBrushSize, glm::vec3(1.0f, 0.0f, 0.0f));     // Red
    drawCircleIndicator(adjustedPosition, updatedIndicatorNormal, scaledBrushSize * 1.5f, glm::vec3(1.0f, 1.0f, 0.0f)); // Yellow

    // Draw center point
    glBindBuffer(GL_ARRAY_BUFFER, pointVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec3), &sculptIndicatorPosition);

    debugShader->use();
    glm::mat4 MVP = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 1000.0f)
        * camera.GetViewMatrix()
        * glm::mat4(1.0f);
    debugShader->setMat4("MVP", MVP);
    debugShader->setVec3("color", glm::vec3(1.0f, 0.0f, 0.0f));

    glBindVertexArray(pointVAO);
    glPointSize(5.0f);
    glDrawArrays(GL_POINTS, 0, 1);

    //  Draw debug normal vector
    glBegin(GL_LINES);
    glColor3f(0.0f, 1.0f, 1.0f);
    glVertex3f(sculptIndicatorPosition.x, sculptIndicatorPosition.y, sculptIndicatorPosition.z);
    glm::vec3 normalEnd = sculptIndicatorPosition + updatedIndicatorNormal * scaledBrushSize * 1.5f;
    glVertex3f(normalEnd.x, normalEnd.y, normalEnd.z);
    glEnd();
}


void updateBrushAreaIndicator() {
    if (!sculptIndicatorVisible) return;

    auto startTotal = high_resolution_clock::now();
    auto startBFS = high_resolution_clock::now();

    // === Step 1: Collect local vertices around the indicator using BFS ===
    std::unordered_set<int> localVertices;
    std::queue<MyMesh::VertexHandle> queue;
    std::vector<bool> visited(mesh.n_vertices(), false);

    float scaledBrushSize = getScaledBrushSize();
    float outerRadius = scaledBrushSize * 1.5f;

    queue.push(sculptIndicatorVertex);
    visited[sculptIndicatorVertex.idx()] = true;

    while (!queue.empty()) {
        MyMesh::VertexHandle vh = queue.front();
        queue.pop();

        localVertices.insert(vh.idx());

        glm::vec3 p = toVec3(mesh.point(vh));
        if (glm::length(p - sculptIndicatorPosition) > outerRadius * 2.0f) continue;

        for (auto vv_it = mesh.vv_iter(vh); vv_it.is_valid(); ++vv_it) {
            if (!visited[vv_it->idx()]) {
                visited[vv_it->idx()] = true;
                queue.push(*vv_it);
            }
        }
    }

    auto endBFS = high_resolution_clock::now();
    auto startSample = high_resolution_clock::now();

    // === Step 2: Sample points around the indicator ===
    glm::vec3 forward = camera.Front;
    glm::vec3 up = glm::abs(forward.z) > 0.99f ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 right = glm::normalize(glm::cross(forward, up));
    glm::vec3 localUp = glm::cross(right, forward);

    glm::vec3 normalSum(0.0f);
    int validSamples = 0;

    const int sampleCount = 8;
    // === Step 2a: Precompute nearby face normals (just once) ===
    std::vector<glm::vec3> faceNormals;
    std::vector<float> faceWeights;

    if (!faceCacheValid) {
        filterLocalFaceCache(sculptIndicatorPosition, outerRadius);
    }



    for (size_t i = 0; i < faceCenterCache.size(); ++i) {
        if (glm::length(faceCenterCache[i] - sculptIndicatorPosition) > outerRadius)
            continue;

        normalSum += faceNormalCache[i] * faceAreaCache[i];
        validSamples++;
    }



    auto endSample = high_resolution_clock::now();
    auto startNormalize = high_resolution_clock::now();

    // === Step 3: Normalize final indicator normal ===
    if (validSamples > 0) {
        updatedIndicatorNormal = glm::normalize(normalSum);
    } else {
        updatedIndicatorNormal = camera.Front;
    }


    auto endNormalize = high_resolution_clock::now();
    auto endTotal = high_resolution_clock::now();

    // === ⏱️ Debug Timing Logs ===
    /*std::cout << "[updateBrushAreaIndicator] BFS Time: "
        << duration_cast<milliseconds>(endBFS - startBFS).count() << " ms\n";
    std::cout << "[updateBrushAreaIndicator] Sampling Time: "
        << duration_cast<milliseconds>(endSample - startSample).count() << " ms\n";
    std::cout << "[updateBrushAreaIndicator] Normalize Time: "
        << duration_cast<microseconds>(endNormalize - startNormalize).count() << " μs\n";
    std::cout << "[updateBrushAreaIndicator] Total Time: "
        << duration_cast<milliseconds>(endTotal - startTotal).count() << " ms\n";*/
}



void drawLightSource() {
    glUseProgram(0); // Disable shaders for immediate mode rendering

    glPointSize(10.0f); // Bigger point for visibility
    glBegin(GL_POINTS);
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow color for light source
    glm::vec3 lightPos = camera.Position + glm::vec3(0.0f, 2.5f, 0.0f);
    glVertex3f(lightPos.x, lightPos.y, lightPos.z);
    glEnd();
}

void renderMesh() {
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    shaderProgram->use();

    // Matrices
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 1000.0f);

    // **Move Point Light Above the Camera** (Follow Camera Movement)
    glm::vec3 lightOffset(0.0f, 1.0f, 0.0f); // Light is always 2.5 units above the camera
    glm::vec3 lightPos = camera.Position + lightOffset;

    // Send updated light position to the shader
    shaderProgram->setVec3("lightPos", lightPos);
    shaderProgram->setVec3("lightColor", 1.0f, 1.0f, 1.0f);
    shaderProgram->setVec3("viewPos", camera.Position);


    // Set transformation matrices
    shaderProgram->setMat4("model", model);
    shaderProgram->setMat4("view", view);
    shaderProgram->setMat4("projection", projection);

    // Set object color
    shaderProgram->setVec3("objectColor", 0.6f, 0.6f, 0.6f);

    // **Enable Wireframe Mode if toggled**
    if (wireframeMode)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Draw the mesh
    glBindVertexArray(VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    // **Restore Solid Mode**
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

GLuint pickingFBO, pickingTexture, pickingDepth;

void setupPickingFramebuffer() {
    glGenFramebuffers(1, &pickingFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenTextures(1, &pickingTexture);
    glBindTexture(GL_TEXTURE_2D, pickingTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glGenRenderbuffers(1, &pickingDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, pickingDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pickingTexture, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, pickingDepth);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "Picking framebuffer not complete!\n";

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void renderForPicking() {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    pickingShader->use();

    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 1000.0f);
    glm::mat4 MVP = projection * view * model;
    pickingShader->setMat4("MVP", MVP);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  /*  GLint isAttribEnabled;
    glGetVertexAttribiv(2, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &isAttribEnabled);
    std::cout << "Attr 2 enabled? " << isAttribEnabled << std::endl;*/
    glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

}


//bool needsPicking = false; // ✅ Only trigger when picking is needed

void pickVertex(GLFWwindow* window) {
    if (isSnakeHookDragging) return;
   // if (!needsPicking) return; // Minimum interval between picks
    static double lastPicking = 0;
    double now = glfwGetTime();
   
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    renderForPicking();
    glFlush();
    glFinish();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    unsigned char data[4];
  glReadPixels((int)lastX, SCR_HEIGHT - (int)lastY, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);


    int pickedID = data[0] + data[1] * 256 + data[2] * 256 * 256;


    unsigned int vertexMappingIdx = pickedID * 3;
    if (pickedID == 0x00ffffff) {
        sculptIndicatorVisible = false;
       // needsPicking = false;
        return;
    }
	if (pickedID >= vertexHandleMapping.size() / 3) {
		//std::cout << "Invalid pick ID: " << pickedID << std::endl;
		sculptIndicatorVisible = false;
		return;
	}

    sculptIndicatorVertex = vertexHandleMapping[vertexMappingIdx];

    MyMesh::Point p = mesh.point(sculptIndicatorVertex);
    sculptIndicatorPosition = glm::vec3(p[0], p[1], p[2]);

    MyMesh::Normal vertexNormal = mesh.normal(sculptIndicatorVertex);
    sculptIndicatorNormal = glm::normalize(glm::vec3(vertexNormal[0], vertexNormal[1], vertexNormal[2]));

    static int frameNumber = 0;
    //std::cout << "Picked vertex: " << sculptIndicatorVertex.idx() << "frame:" << frameNumber++ << " at position : " << sculptIndicatorPosition.x << ", " << sculptIndicatorPosition.y << ", " << sculptIndicatorPosition.z << std::endl;
    sculptIndicatorVisible = true;


    lastPicking = now;
}

void updateAffectedTriangles(const std::unordered_set<int>& vertexIDs) {
    std::unordered_set<size_t> affectedTriangles;

    for (int vid : vertexIDs) {
        for (size_t triIndex : vertexToTriangleMap[vid]) {
            affectedTriangles.insert(triIndex);
        }
    }

    for (size_t triIndex : affectedTriangles) {
        size_t base = triIndex * 3;
        glm::vec3 v0 = toVec3(mesh.point(vertexHandleMapping[base + 0]));
        glm::vec3 v1 = toVec3(mesh.point(vertexHandleMapping[base + 1]));
        glm::vec3 v2 = toVec3(mesh.point(vertexHandleMapping[base + 2]));

        glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));

        std::array<glm::vec3, 3> points = { v0, v1, v2 };

        for (int i = 0; i < 3; ++i) {
            size_t idx = triIndex * 18 + i * 6;
            vertices[idx + 0] = points[i].x;
            vertices[idx + 1] = points[i].y;
            vertices[idx + 2] = points[i].z;
            vertices[idx + 3] = faceNormal.x;
            vertices[idx + 4] = faceNormal.y;
            vertices[idx + 5] = faceNormal.z;
        }
    }
}


void sculptMesh(float sculptStrength, bool isClay) {
    if (!sculptIndicatorVisible) return;
    auto startTotal = high_resolution_clock::now();

    float scaledBrushSize = getScaledBrushSize();
    float extendedBrushSize = scaledBrushSize * 5.0f;

    int count = 0;

    std::queue<std::pair<MyMesh::VertexHandle, float>> vertexQueue;
    std::vector<bool> visitedFlags(mesh.n_vertices(), false);
    visitedFlags[sculptIndicatorVertex.idx()] = true;
    vertexQueue.push({ sculptIndicatorVertex, 0.0f });

    auto startBFS = high_resolution_clock::now();

    std::unordered_set<int> visitedVertices;
    std::unordered_set<size_t> affectedTriangles;
    std::unordered_map<int, float> vertexInfluence;

    // BFS loop
    while (!vertexQueue.empty()) {
        auto frontPair = vertexQueue.front();
        vertexQueue.pop();

        MyMesh::VertexHandle current = frontPair.first;
        float currentDist = frontPair.second;

        if (!mesh.is_valid_handle(current)) continue;

        glm::vec3 vertexPos = toVec3(mesh.point(current));
        float planarDistance = glm::length(vertexPos - sculptIndicatorPosition);

        if (planarDistance < extendedBrushSize) {
            float influence = pow(std::max(0.0f, gaussian(planarDistance, scaledBrushSize)), 2.5f);
            visitedVertices.insert(current.idx());
            vertexInfluence[current.idx()] = influence;
            count++;

            for (auto vv_it = mesh.vv_iter(current); vv_it.is_valid(); ++vv_it) {
                if (!mesh.is_valid_handle(*vv_it)) continue;
                if (!visitedFlags[vv_it->idx()]) {
                    visitedFlags[vv_it->idx()] = true;
                    float dist = currentDist + glm::distance(vertexPos, toVec3(mesh.point(*vv_it)));
                    if (dist <= extendedBrushSize) {
                        vertexQueue.push({ *vv_it, dist });
                    }
                }
            }
        }
    }
    auto endBFS = high_resolution_clock::now();

    auto startApply = high_resolution_clock::now();

    // 🛠️ 2. Apply Sculpting after BFS 
    // FIX::USE VERTEX NORMAL
    glm::vec3 sculptDir = isClay ? -camera.Front : camera.Front;
    for (int vid : visitedVertices) {
        MyMesh::VertexHandle vh = MyMesh::VertexHandle(vid);
        if (!mesh.is_valid_handle(vh)) continue;

        glm::vec3 vertexPos = toVec3(mesh.point(vh));
        float influence = vertexInfluence[vid];
        glm::vec3 newPosition = vertexPos + (sculptDir * sculptStrength * influence);

        mesh.set_point(vh, MyMesh::Point(newPosition.x, newPosition.y, newPosition.z));
    }

    auto endApply = high_resolution_clock::now();



    auto startNormalUpdate = high_resolution_clock::now();

    updateAffectedTriangles(visitedVertices);

    auto endNormalUpdate = high_resolution_clock::now();

    meshDirty = true;
    faceCacheValid = false;
    MyMesh::Point updated = mesh.point(sculptIndicatorVertex);
    sculptIndicatorPosition = glm::vec3(updated[0], updated[1], updated[2]);

    auto endTotal = high_resolution_clock::now();

    // ⏱️ Output Timings
    std::cout << "Sculpted " << visitedVertices.size() << " vertices.\n";
    std::cout << "Sculpted " << affectedTriangles.size() << " triangles.\n";
    std::cout << "[sculptMesh] BFS Time: "
        << duration_cast<milliseconds>(endBFS - startBFS).count() << " ms\n";
    std::cout << "[sculptMesh] Sculpt Apply Time: " 
        << duration_cast<milliseconds>(endApply - startApply).count() << " ms\n";
    std::cout << "[sculptMesh] Normal Update Time: "
        << duration_cast<milliseconds>(endNormalUpdate - startNormalUpdate).count() << " ms\n";
    std::cout << "[sculptMesh] Total Time: "
        << duration_cast<milliseconds>(endTotal - startTotal).count() << " ms\n";
}

void pullVertexAlongView(MyMesh::VertexHandle vertex, const glm::vec2& startScreen, const glm::vec3& startWorld) {
    glm::vec2 screenDelta = glm::vec2(lastX, lastY) - startScreen;
    float strength = screenDelta.y * 0.01f; // Inverted to match expected pull
    glm::vec3 pullDir = glm::normalize(camera.Front);
    glm::vec3 newPos = startWorld + pullDir * strength;

    mesh.set_point(vertex, MyMesh::Point(newPos.x, newPos.y, newPos.z));
    sculptIndicatorPosition = newPos;
    // Update vertex buffer for affected triangles
    std::unordered_set<int> affectedVertexIDs = { vertex.idx() };
    updateAffectedTriangles(affectedVertexIDs);

    meshDirty = true;
    faceCacheValid = false; // In case normals/area are cached for brush
}


void updateMeshBuffers_FullUpload() {
    using namespace std::chrono;
    auto startTotal = high_resolution_clock::now();

    auto startUpload = high_resolution_clock::now();

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);

    auto endUpload = high_resolution_clock::now();

    std::cout << "[updateMeshBuffers_FullUpload] glBufferData Time: "
        << duration_cast<milliseconds>(endUpload - startUpload).count() << " ms\n";

    auto endTotal = high_resolution_clock::now();
    std::cout << "[updateMeshBuffers_FullUpload] Total Function Time: "
        << duration_cast<milliseconds>(endTotal - startTotal).count() << " ms\n";
}



void splitLongEdgesInBrush(const glm::vec3& center, float radius, float maxEdgeLength = 0.25f) {
    std::vector<MyMesh::EdgeHandle> toSplit;

    for (auto e_it = mesh.edges_begin(); e_it != mesh.edges_end(); ++e_it) {
        if (!mesh.is_valid_handle(*e_it)) continue;

        auto heh = mesh.halfedge_handle(*e_it, 0);
        auto from = mesh.from_vertex_handle(heh);
        auto to = mesh.to_vertex_handle(heh);

        glm::vec3 p0(mesh.point(from)[0], mesh.point(from)[1], mesh.point(from)[2]);
        glm::vec3 p1(mesh.point(to)[0], mesh.point(to)[1], mesh.point(to)[2]);

        glm::vec3 midpoint = 0.5f * (p0 + p1);
        float dist = glm::length(midpoint - center);

        if (dist < radius && glm::length(p1 - p0) > maxEdgeLength) {
            toSplit.push_back(*e_it);
        }
    }

    for (const auto& edge : toSplit) {
        if (!mesh.is_valid_handle(edge)) continue;

        auto heh = mesh.halfedge_handle(edge, 0);
        auto from = mesh.from_vertex_handle(heh);
        auto to = mesh.to_vertex_handle(heh);

        glm::vec3 p0(mesh.point(from)[0], mesh.point(from)[1], mesh.point(from)[2]);
        glm::vec3 p1(mesh.point(to)[0], mesh.point(to)[1], mesh.point(to)[2]);

        glm::vec3 mid = 0.5f * (p0 + p1);
        MyMesh::VertexHandle newV = mesh.add_vertex(MyMesh::Point(mid.x, mid.y, mid.z));

        mesh.split(edge, newV);
    }

    mesh.update_normals();
}

double lastPickTime = 0.0;
const double PICK_INTERVAL = 0.01; // Only pick every 50ms (~20 FPS)
bool draggedThisFrame = false;

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    if (snakeHookMode && isSnakeHookDragging) {
        currentDragPos = glm::vec2(lastX, lastY); // Keep updating current drag position
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;  // reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    if ((isSculptingUp || isSculptingDown) && (std::abs(xoffset) > 0.5f || std::abs(yoffset) > 0.5f)) {
        draggedThisFrame = true;
    }


    if (snakeHookMode && isSnakeHookDragging) {
        currentDragPos = glm::vec2(lastX, lastY);
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
        isOrbiting = true;

        const float sensitivity = 0.1f;
        yaw += xoffset * sensitivity;
        pitch -= yoffset * sensitivity;

        pitch = std::max(-89.0f, std::min(89.0f, pitch));

        glm::vec3 newCameraPos;
        newCameraPos.x = orbitCenter.x + distance * cos(glm::radians(pitch)) * cos(glm::radians(yaw));
        newCameraPos.y = orbitCenter.y + distance * sin(glm::radians(pitch));
        newCameraPos.z = orbitCenter.z + distance * cos(glm::radians(pitch)) * sin(glm::radians(yaw));

        camera.Position = newCameraPos;
        camera.Front = glm::normalize(orbitCenter - camera.Position);
    }
    else {
        isOrbiting = false;
    }
}


const float MIN_BRUSH_SIZE = 0.05f;
const float MAX_BRUSH_SIZE = 2.0f;  // 🔹 Set the maximum brush size

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
        // Ctrl + scroll changes brush size
        brushSize += yoffset * 0.02f;
        brushSize = std::max(MIN_BRUSH_SIZE, std::min(brushSize, MAX_BRUSH_SIZE));

        std::cout << "Brush Size (Ctrl+Scroll): " << brushSize << " | Scaled: " << getScaledBrushSize() << std::endl;
    }
    else if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        // Shift + scroll changes brush strength
        brushStrength += yoffset * 0.002f;
        brushStrength = std::max(MIN_BRUSH_STRENGTH, std::min(brushStrength, MAX_BRUSH_STRENGTH));

        std::cout << "Brush Strength (Shift+Scroll): " << brushStrength << std::endl;
    }
    else {
        // Normal scroll zooms the camera
        distance -= yoffset * 0.2f;
        distance = std::max(1.0f, std::min(distance, 20.0f));

        glm::vec3 newCameraPos;
        newCameraPos.x = orbitCenter.x + distance * cos(glm::radians(pitch)) * cos(glm::radians(yaw));
        newCameraPos.y = orbitCenter.y + distance * sin(glm::radians(pitch));
        newCameraPos.z = orbitCenter.z + distance * cos(glm::radians(pitch)) * sin(glm::radians(yaw));

        camera.Position = newCameraPos;
        camera.Front = glm::normalize(orbitCenter - camera.Position);

        std::cout << "Camera Distance: " << distance << std::endl;
    }
}



void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if ((button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) && action == GLFW_PRESS && !snakeHookMode) {
        isSculptingUp = (button == GLFW_MOUSE_BUTTON_LEFT);
        isSculptingDown = (button == GLFW_MOUSE_BUTTON_RIGHT);
       // isDraggingWhileSculpting = false; // reset drag tracking
    }
    if ((button == GLFW_MOUSE_BUTTON_LEFT || button == GLFW_MOUSE_BUTTON_RIGHT) && action == GLFW_RELEASE) {
        isSculptingUp = false;
        isSculptingDown = false;
       // isDraggingWhileSculpting = false;
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && isPulling) {
        if (sculptIndicatorVisible) {
            pullActive = true;
            pullVertex = sculptIndicatorVertex;
            pullStartWorldPos = sculptIndicatorPosition;
            pullStartScreenPos = glm::vec2(lastX, lastY);
        }
    }

    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        pullActive = false;
    }


    if (action == GLFW_RELEASE) {
        isSnakeHookDragging = false;
        lockSculptIndicator = false;
    }
}



#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#else
#include <unistd.h>
#include <limits.h>
#endif

void exportMeshToOBJ(const std::string& filename) {
    if (!OpenMesh::IO::write_mesh(mesh, filename)) {
        std::cerr << " Error: Failed to export mesh to " << filename << "\n";
        return;
    }

    std::cout << "Successfully exported obj to: ";

#ifdef _WIN32
    char fullPath[MAX_PATH];
    if (_fullpath(fullPath, filename.c_str(), MAX_PATH)) {
        std::cout << fullPath << "\n";
    }
    else {
        std::cout << filename << " (failed to get absolute path)\n";
    }
#else
    char fullPath[PATH_MAX];
    if (realpath(filename.c_str(), fullPath)) {
        std::cout << fullPath << "\n";
    }
    else {
        std::cout << filename << " (failed to get absolute path)\n";
    }
#endif
}

void processInput(GLFWwindow* window) {
    float currentFrame = static_cast<float>(glfwGetTime());
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        exportMeshToOBJ("sculpture.obj");

    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS)
        wireframeMode = !wireframeMode;


    static bool fKeyLast = false;
    bool fKeyNow = glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS;

    if (fKeyNow && !fKeyLast) {
        if (sculptIndicatorVisible) {
            isPulling = !isPulling;
            std::cout << "Pull mode: " << (isPulling ? "ON" : "OFF") << std::endl;
        }
    }
    fKeyLast = fKeyNow;


    // ✅ Prevent sculpting if pull is active
    if (!isPulling) {
        if (isSculptingUp && draggedThisFrame) {
            sculptMesh(brushStrength, false);
        }
        if (isSculptingDown && draggedThisFrame) {
            sculptMesh(brushStrength, true);
        }
    }

    if (isPulling && pullActive && mesh.is_valid_handle(pullVertex)) {
        pullVertexAlongView(pullVertex, pullStartScreenPos, pullStartWorldPos);
    }



    // ✅ Brush size adjustment with arrow keys
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        brushSize += 0.01f;
        brushSize = std::min(brushSize, MAX_BRUSH_SIZE);
        std::cout << "Brush Size: " << brushSize << " | Scaled: " << getScaledBrushSize() << std::endl;
    }

    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        brushSize -= 0.01f;
        brushSize = std::max(brushSize, MIN_BRUSH_SIZE);
        std::cout << "Brush Size: " << brushSize << " | Scaled: " << getScaledBrushSize() << std::endl;
    }

    static bool upPressedLast = false;
    static bool downPressedLast = false;

    bool upPressed = glfwGetKey(window, GLFW_KEY_PAGE_UP) == GLFW_PRESS;
    bool downPressed = glfwGetKey(window, GLFW_KEY_PAGE_DOWN) == GLFW_PRESS;

    if (upPressed && !upPressedLast && currentSubdivisionLevel < maxSubdivision) {
        subdivideMesh(1);                      // Only one level at a time
        currentSubdivisionLevel += 1;
    }

    if (downPressed && !downPressedLast && currentSubdivisionLevel > 0) {
       
        std::cout << "[WARN] Cannot revert subdivision: no undo history implemented.\n";
     
    }
    
    upPressedLast = upPressed;
    downPressedLast = downPressed;

    draggedThisFrame = false;

 
}


int main() {
    if (!glfwInit()) {
        std::cerr << "Error: Failed to initialize GLFW.\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Sculpting Viewer", NULL, NULL);
   
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);

    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);

    generateIcosphere(currentSubdivisionLevel, 1.5f);
   
    shaderProgram = new Shader("vertex_shader.vert", "fragment_shader.frag");
    pickingShader = new Shader("pick_shader.vert", "pick_shader.frag");
    debugShader = new Shader("debug.vert", "debug.frag");

    setupMeshBuffers();
    setupPickingFramebuffer();
	setupIndicatorframebuffer();
    precomputeAllFaceData();

    frameCount = 0;
    previousTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glfwPollEvents();

        double currentTime = glfwGetTime();
       

        if (meshDirty) {
            updateMeshBuffers_FullUpload();
            meshDirty = false;
        }

        if (!pullActive && !isSnakeHookDragging) {
            pickVertex(window);
        }

        renderMesh();
        updateBrushAreaIndicator();
        drawSculptIndicatorAndBrushArea();
		

        glfwSwapBuffers(window);

        // FPS display logic
        frameCount++;

        if (currentTime - previousTime >= 1.0) {
            double fps = double(frameCount) / (currentTime - previousTime);

            std::string title = "Sculpting Viewer - FPS: " + std::to_string(int(fps));
            glfwSetWindowTitle(window, title.c_str());

            frameCount = 0;
            previousTime = currentTime;
        }
    }

    delete shaderProgram;
    glfwTerminate();
    return 0;
}
