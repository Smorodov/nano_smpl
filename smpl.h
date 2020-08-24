#include "cnpy.h"
#include <iostream>
#include <queue> // очередь
#include <stack> // стек
#include <vector>
#include <set>
#include <omp.h>
#define COUT_VAR(x) std::cout << #x"=" << x << std::endl;
#define SHOW_IMG(x) cv::namedWindow(#x);cv::imshow(#x,x);cv::waitKey(20);
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
class smpl
{
    //---------------------------------------------------------------------------
    // Simple class for triangle mesh normals computarion
    //---------------------------------------------------------------------------
    class NormalsComputer
    {
    private:
        // vector of vertices shared triangles
        // vertex normals compured by averaging face normals
        std::vector < std::set<unsigned int> > sharedTriangles;
        std::vector<unsigned int> faces;
    public:
        //-------------------
        // constructor
        //-------------------
        NormalsComputer(std::vector<unsigned int>& faces)
        {
            this->faces.assign(faces.begin(), faces.end());
            // need to compute once per mesh
            computeSharedTriangles(faces, sharedTriangles);
        }
        //-------------------
        // destructor
        //-------------------
        ~NormalsComputer()
        {

        }
        //--------------------------------------------------------------------------
        // compute vertex normals
        // input: vertices as array of floats in v1x,v1y,v1z,...vmx,vmy,vmz format
        //--------------------------------------------------------------------------
        void getVertexNormals(unsigned int nVertices, float* vertices, float* verticesNormals)
        {
            computeVertexNormals(faces.size(), nVertices, faces.data(), vertices, verticesNormals);
        }
        //--------------------------------------------------------------------------
        // same for std::vector type 
        //--------------------------------------------------------------------------
        void getVertexNormals(std::vector<float>& vertices, std::vector<float>& verticesNormals)
        {
            computeVertexNormals(faces, vertices, verticesNormals);
        }
        //--------------------------------------------------------------------------
        // same for face normals
        //--------------------------------------------------------------------------
        void getFaceNormals(std::vector<float>& vertices, std::vector<float>& normals)
        {
            computeFaceNormals(faces, vertices, normals);
        }
    private:
        //--------------------------------------------------------------------------
        // compute trianges shared by each vertex
        //--------------------------------------------------------------------------
        void computeSharedTriangles(std::vector<unsigned int>& faces, std::vector < std::set<unsigned int> >& result)
        {
            unsigned int max_ind = *std::max_element(faces.begin(), faces.end()) + 1;
            result.resize(max_ind);

            for (unsigned int i = 0; i < faces.size() / 3; ++i)
            {
                unsigned int a = faces.data()[i * 3 + 0];
                unsigned int b = faces.data()[i * 3 + 1];
                unsigned int c = faces.data()[i * 3 + 2];

                if (a == b || a == c || b == c)
                {
                    std::cout << "Incorrect triangle !" << std::endl;
                }
                result[a].insert(i);
                result[b].insert(i);
                result[c].insert(i);
            }
        }
        //--------------------------------------------------------------------------
        // compute one face normsl
        //--------------------------------------------------------------------------
        void computeNormal(float* a, float* b, float* c, float* normal)
        {
            float v1[3], v2[3];
            v1[0] = b[0] - a[0];
            v1[1] = b[1] - a[1];
            v1[2] = b[2] - a[2];

            v2[0] = b[0] - c[0];
            v2[1] = b[1] - c[1];
            v2[2] = b[2] - c[2];

            normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
            normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
            normal[2] = v1[0] * v2[1] - v1[1] * v2[0];
            float nx = normal[0];// / norm;
            float ny = normal[1];// / norm;
            float nz = normal[2];// / norm;
            float norm = sqrt(nx * nx + ny * ny + nz * nz);
            //
            normal[0] = -nx;
            normal[1] = -ny;
            normal[2] = -nz;
        }
        //--------------------------------------------------------------------------
        // compute all face normals
        //--------------------------------------------------------------------------
        void computeFaceNormals(std::vector<unsigned int>& faces, std::vector<float>& vertices, std::vector<float>& normals)
        {
            normals.clear();
            normals.resize(faces.size(), 0);
#pragma omp parallel for
            for (unsigned int i = 0; i < faces.size() / 3; ++i)
            {
                float* a;
                float* b;
                float* c;
                float* N;
                unsigned int a_ind = faces[3 * i + 0];
                unsigned int b_ind = faces[3 * i + 1];
                unsigned int c_ind = faces[3 * i + 2];
                if (a_ind == b_ind || a_ind == c_ind || b_ind == c_ind)
                {
                    std::cout << "Incorrect triangle !" << std::endl;
                }
                a = vertices.data() + a_ind * 3;
                b = vertices.data() + b_ind * 3;
                c = vertices.data() + c_ind * 3;
                computeNormal(a, b, c, normals.data() + i * 3);
            }
        }
        //--------------------------------------------------------------------------
        // compute vertex normals by averaging shared face normals
        //--------------------------------------------------------------------------
        void computeVertexNormals(std::vector<unsigned int>& faces, std::vector<float>& vertices, std::vector<float>& result)
        {
            result.resize(vertices.size());
            std::fill(result.begin(), result.end(), 0);
            std::vector<float> faceNormals;
            computeFaceNormals(faces, vertices, faceNormals);
            for (unsigned int i = 0; i < sharedTriangles.size(); ++i)
            {
                std::set<unsigned int>::iterator it;
                for (it = sharedTriangles[i].begin(); it != sharedTriangles[i].end(); ++it)
                {
                    unsigned int t = *it;
                    result[i * 3 + 0] += faceNormals[t * 3 + 0];
                    result[i * 3 + 1] += faceNormals[t * 3 + 1];
                    result[i * 3 + 2] += faceNormals[t * 3 + 2];
                }
                float n = sqrt(result[i * 3 + 0] * result[i * 3 + 0] + result[i * 3 + 1] * result[i * 3 + 1] + result[i * 3 + 2] * result[i * 3 + 2]);
                if (n > 0)
                {
                    result[i * 3 + 0] /= n;
                    result[i * 3 + 1] /= n;
                    result[i * 3 + 2] /= n;
                }
                else
                {
                    std::cout << "zero normal" << std::endl;
                }
            }
        }
        //--------------------------------------------------------------------------
        // smae as before for pointers
        //--------------------------------------------------------------------------
        void computeVertexNormals(unsigned int nFaces, unsigned int nVertices, unsigned int* faces, float* vertices, float* result)
        {
            std::vector<unsigned int> vfaces(nFaces);
            std::vector<float> vvertices(nVertices);
            memcpy(vfaces.data(), faces, nFaces * sizeof(unsigned int));
            memcpy(vvertices.data(), vertices, nVertices * sizeof(float));
            std::vector<float> vresult;
            computeVertexNormals(vfaces, vvertices, vresult);
            memcpy(result, vresult.data(), vresult.size() * sizeof(float));
        }
    };    
    //--------------------------------------------------------------------------
     //
     //--------------------------------------------------------------------------
    class Skeleton
    {
    public:
        int root_ind;
        std::vector<int> l1;// = { -1,0,1,2,0,4,5,0,7,8,0,10,11,0,13,14 };
        std::vector<int> l2;// = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
        std::vector<int> level;
        std::vector<int> has_child;
        std::vector < std::vector<int> > paths;
        std::vector< std::vector<int> > mas;
        std::vector < std::vector<int> > chains;

        std::vector<float> joints_rots_vector;
        cnpy::NpyArray template_joints_abs_coords;        
        cnpy::NpyArray posed_joints_abs_coords;
        cnpy::NpyArray template_joints_rel_coords;
        cnpy::NpyArray posed_joints_rel_coords;
        float root_transform[16];
        cnpy::NpyArray joint_rel_transforms;
        cnpy::NpyArray joint_abs_transforms;
        // thit need to apply pose diiven pca shape deformations
        cnpy::NpyArray pose_maps;
        cnpy::NpyArray hands_mean;
        //-----------------------------------
        // skeleton
        //-----------------------------------
        struct Edge
        {
            int begin;
            int end;
        };
        //-----------------------------------
        // constructor
        //-----------------------------------
        Skeleton();
        void parse(cnpy::NpyArray& Kinematic_tree);
        void setPose(std::vector<float>& joints_rots_vector);
    };
public:
    int face_index_num;
    int vertex_num;
    int shape_basis_dim;
    int pose_basis_dim;
    int joint_num;
    cnpy::npz_t npz_map;
    cnpy::NpyArray kinematicTree;
    cnpy::NpyArray faceIndices;
    cnpy::NpyArray vertices_template;
    cnpy::NpyArray shapeBlendBasis;
    cnpy::NpyArray poseBlendBasis;
    cnpy::NpyArray jointRegressor;
    cnpy::NpyArray weights;
    // used for hand model only
    cnpy::NpyArray hands_components;
    cnpy::NpyArray hands_coeffs;    
    // ----------------------------
    // for MANO_left.npz should be:
    //face_index_num=1538
    //vertex_num=778
    //shape_basis_dim=10
    //pose_basis_dim=135
    //joint_num=16
    //kinematicTree shape=[2,16]
    //faceIndices shape [1538,3]
    //vertices_template shape:[778,3]
    //shapeBlendBasis shape:[778,3,10]
    //poseBlendBasis shape:[778,3,135]
    //jointRegressor shape:[16,778]
    //weights shape:[778,16]
    // ----------------------------
    cnpy::NpyArray Beta;
    cnpy::NpyArray v_shaped;
    cnpy::NpyArray v_posed;
    cnpy::NpyArray posed_vertices;
    cnpy::NpyArray vetexNormals;
    cnpy::NpyArray jointCoords;
    Skeleton skeleton;
    NormalsComputer* normalsComputer;
    smpl();
    ~smpl();
public:
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void loadModel(std::string filename);
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void computeBlendShape(cnpy::NpyArray& Beta, cnpy::NpyArray& MeanShape, cnpy::NpyArray& ShapeBlendBasis, cnpy::NpyArray& V_Shaped);
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------

    void smpl::computeBlendPose(cnpy::NpyArray& PoseMap, cnpy::NpyArray& V_Shaped, cnpy::NpyArray& PoseBlendBasis, cnpy::NpyArray& V_Posed);
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void BlendTransforms(cnpy::NpyArray& weights, cnpy::NpyArray& joint_rel_transforms, cnpy::NpyArray& blendedTransforms);
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void LinearBlend(cnpy::NpyArray& template_vertices, cnpy::NpyArray& weights, cnpy::NpyArray& joint_rel_transforms, cnpy::NpyArray& posed_vertices);
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void computeJointsFromShape(cnpy::NpyArray& Shape, cnpy::NpyArray& JointRegressor, cnpy::NpyArray& JointCoords);
    //--------------------------------------------------------------------------
    //
    //--------------------------------------------------------------------------
    void updateMesh(std::vector<float>& shape, std::vector<float>& psoe);
};
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
