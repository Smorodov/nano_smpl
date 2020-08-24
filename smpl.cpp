// http://rodolphe-vaillant.fr/?e=29

#include "smpl.h"
#include <iomanip>
#include <chrono>
#include <xmmintrin.h>
// http://web.archive.org/web/20150531202539/http://www.codeproject.com/Articles/4522/Introduction-to-SSE-Programming
// https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=SSE2
#include <omp.h>
// --------------------------------------
//
// --------------------------------------
void print3dArray(cnpy::NpyArray& arr)
{
    std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
    for (size_t i = 0; i < arr.shape[0]; i++)
    {
        for (size_t j = 0; j < arr.shape[1]; j++)
        {
            for (size_t k = 0; k < arr.shape[2]; k++)
            {
                std::cout << std::setw(8) << arr.data<float>()[i * arr.shape[1] * arr.shape[2] + j * arr.shape[2] + k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << " -------- " << std::endl;
    }
    std::cout << " ================= " << std::endl;
}
// --------------------------------------
//
// --------------------------------------
void print2dArray(cnpy::NpyArray& arr)
{
    std::cout << std::fixed << std::setprecision(4) << std::setfill(' ');
    for (size_t i = 0; i < arr.shape[0]; i++)
    {
        for (size_t j = 0; j < arr.shape[1]; j++)
        {
            std::cout << std::setw(8) << arr.data<float>()[i * arr.shape[1] + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << " ================= " << std::endl;
}
// --------------------------------------
//
// --------------------------------------
void quat2mat(float* quat, float* rotMat)
{
    //Convert quaternion coefficients to rotation matrix.
    //Args:
    //    quat: w, x, y, z
    //Returns:
    //    Rotation matrix corresponding to the quaternion -- size = 3x3

    float norm_quat[4];
    float l2_norm = 0;
    for (int i = 0; i < 4; ++i)
    {
        l2_norm += quat[i] * quat[i];
    }
    l2_norm = sqrt(l2_norm);

    for (int i = 0; i < 4; ++i)
    {
        if (l2_norm > 0)
        {
            norm_quat[i] = quat[i] / l2_norm;
        }
        else
        {
            norm_quat[i] = 0;
        }
    }
    float w = norm_quat[0];
    float x = norm_quat[1];
    float y = norm_quat[2];
    float z = norm_quat[3];

    float w2 = norm_quat[0] * norm_quat[0];
    float x2 = norm_quat[1] * norm_quat[1];
    float y2 = norm_quat[2] * norm_quat[2];
    float z2 = norm_quat[3] * norm_quat[3];

    float wx = w * norm_quat[1];
    float wy = w * norm_quat[2];
    float wz = w * norm_quat[3];

    float xy = norm_quat[1] * norm_quat[2];
    float xz = norm_quat[1] * norm_quat[3];
    float yz = norm_quat[2] * norm_quat[3];

    rotMat[0] = w2 + x2 - y2 - z2;
    rotMat[1] = 2 * (xy - wz);
    rotMat[2] = 2 * (wy + xz);
    rotMat[3] = 2 * (wz + xy);
    rotMat[4] = w2 - x2 + y2 - z2;
    rotMat[5] = 2 * (yz - wx);
    rotMat[6] = 2 * (xz - wy);
    rotMat[7] = 2 * (wx + yz);
    rotMat[8] = w2 - x2 - y2 + z2;
}
// --------------------------------------
//
// --------------------------------------
void rodrigues(float* axisang, float* rot_mat)
{
    // axisang size 3
    // rot_mat size 3x3
    float angle = 0;

    for (int i = 0; i < 3; ++i)
    {
        angle += axisang[i] * axisang[i];
    }
    angle = sqrt(angle);

    float axisang_normalized[3];
    for (int i = 0; i < 3; ++i)
    {
        if (angle > 0)
        {
            axisang_normalized[i] = axisang[i] / angle;
        }
        else
        {
            axisang_normalized[i] = 0;
        }
    }
    angle = angle * 0.5;
    float v_cos = cos(angle);
    float v_sin = sin(angle);
    float quat[4] = { v_cos,v_sin * axisang_normalized[0],v_sin * axisang_normalized[1],v_sin * axisang_normalized[2] };
    quat2mat(quat, rot_mat);
}
// --------------------------------------
// Combine rotation and translation to single transform
// --------------------------------------
void makeTrasformsMatrix(cnpy::NpyArray& rotations, cnpy::NpyArray& translations, cnpy::NpyArray& transforms)
{
    size_t n_transfarms = rotations.shape[0];
    // shape of the result tensors
    std::vector<size_t> result_sz = { n_transfarms,4,4};
    // result tensors
    transforms = cnpy::NpyArray(result_sz, sizeof(float), false);
    // fille tthem with zeros
    memset(transforms.data<float>(), 0, n_transfarms* 16 * sizeof(float));
    // iterare vectors
    for (int n = 0; n < n_transfarms; ++n)
    {
        float* transform = transforms.data<float>() + n * 16;
        float* rotation = rotations.data<float>() + n * 9;
        float* translation = translations.data<float>() + n * 3;
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                size_t ind4x4 = i * 4 + j;
                size_t ind3x3 = i * 3 + j;
                transform[ind4x4] = rotation[ind3x3];
            }
        }
        transform[0 * 4 + 3] = translation[0];
        transform[1 * 4 + 3] = translation[1];
        transform[2 * 4 + 3] = translation[2];
        transform[3 * 4 + 3] = 1;
    }
}
// --------------------------------------
// Convert joints rotations vector to rotation matrices
// --------------------------------------
void posemap_axisang(std::vector<float>& pose_vectors, cnpy::NpyArray& pose_maps, cnpy::NpyArray& rot_mats)
{
    // number of vectors to convert
    size_t rot_nb = int(pose_vectors.size() / 3);
    // shape of the result tensors
    std::vector<size_t> result_sz = { rot_nb,3,3 };
    // result tensors
    pose_maps = cnpy::NpyArray(result_sz, sizeof(float), false);
    rot_mats = cnpy::NpyArray(result_sz, sizeof(float), false);
    // fille tthem with zeros
    memset(pose_maps.data<float>(), 0, 9 * rot_nb * sizeof(float));
    memset(rot_mats.data<float>(), 0, 9 * rot_nb * sizeof(float));
    // iterare vectors
    for (int i = 0; i < rot_nb; ++i)
    {
        float rot_mat[9];
        size_t ind_vec = i * 3;
        float* v = (float*)(pose_vectors.data() + i * 3);
        rodrigues(v, rot_mat);
        for (int j = 0; j < 3; ++j)
        {
#pragma omp parallel for
            for (int k = 0; k < 3; ++k)
            {
                size_t ind4x4 = i * 9 + j * 3 + k;
                size_t ind3x3 = j * 3 + k;
                rot_mats.data<float>()[ind4x4] = rot_mat[ind3x3];
                // pose_maps = rot_maps - eye
                pose_maps.data<float>()[ind4x4] = rot_mats.data<float>()[ind4x4];
                if (j == k)
                {
                    pose_maps.data<float>()[ind4x4] -= 1;
                }
            }
        }
    }
    pose_maps.shape.resize(1);
    pose_maps.shape[0] = result_sz[0] * result_sz[1] * result_sz[2];
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
smpl::Skeleton::Skeleton()
{
    // # finger tips are not joints in MANO, we label them on the mesh manually
    //mesh_mapping = { 16: 333, 17 : 444, 18 : 672, 19 : 555, 20 : 744 }
}
//--------------------------------------------------------------------------
// Clone cnpy::NpyArray
//--------------------------------------------------------------------------
void cloneArray(cnpy::NpyArray& src, cnpy::NpyArray& dst)
{
    dst = cnpy::NpyArray(src.shape, src.word_size, src.fortran_order);
    memcpy(dst.data<unsigned char>(), src.data<unsigned char>(), src.num_bytes());
}
//--------------------------------------------------------------------------
// 
//--------------------------------------------------------------------------
void M4x4_SSE(float* A, float* B, float* C)
{
    __m128 row1 = _mm_load_ps(&B[0]);
    __m128 row2 = _mm_load_ps(&B[4]);
    __m128 row3 = _mm_load_ps(&B[8]);
    __m128 row4 = _mm_load_ps(&B[12]);
    for (int i = 0; i < 4; i++)
    {
        __m128 brod1 = _mm_set1_ps(A[4 * i + 0]);
        __m128 brod2 = _mm_set1_ps(A[4 * i + 1]);
        __m128 brod3 = _mm_set1_ps(A[4 * i + 2]);
        __m128 brod4 = _mm_set1_ps(A[4 * i + 3]);
        __m128 row = _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(brod1, row1),
                _mm_mul_ps(brod2, row2)),
            _mm_add_ps(
                _mm_mul_ps(brod3, row3),
                _mm_mul_ps(brod4, row4)));
        _mm_store_ps(&C[4 * i], row);
    }
}
//--------------------------------------------------------------------------
// 
//--------------------------------------------------------------------------
void matMul4x4(float* A, float* B, float* dst)
{
    const int N = 4;
    int i, j, k;
    bool needBuffer = false;
    if (dst == A || dst == B) { needBuffer = true; }
    float* tmp=nullptr;
    if (needBuffer)
    {
        tmp = new float[N * N];
    }
    else
    {
        tmp = dst;
    }
    
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            tmp[i * N + j] = 0;
            for (k = 0; k < N; k++)
            {
                tmp[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
    if (needBuffer)
    {
        memcpy(dst, tmp, N * N * sizeof(float));
        delete[] tmp;
    }
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void smpl::BlendTransforms(cnpy::NpyArray& weights, cnpy::NpyArray& joint_rel_transforms, cnpy::NpyArray& blendedTransforms)
{
    std::vector<size_t> sz = {weights.shape[0], joint_rel_transforms.shape[1], joint_rel_transforms.shape[2]};
    blendedTransforms = cnpy::NpyArray(sz, joint_rel_transforms.word_size, joint_rel_transforms.fortran_order);
    memset(blendedTransforms.data<float>(), 0, blendedTransforms.num_bytes());
    size_t n_vertices = sz[0];
    size_t transform_sz = sz[1]*sz[2];
    size_t n_bones = weights.shape[1];
    for (size_t i = 0; i < n_vertices; i++) // num vertices 778
    {
        size_t ind1 = i * transform_sz;

#pragma omp parallel for
        for (size_t j = 0; j < n_bones; j++) // num joints 16
        {
            size_t ind2 = j * transform_sz;
            float w = weights.data<float>()[i*weights.shape[1]+j];
            if (fabs(w) > 1e-3) // shrink weak weights
            {
                for (size_t k = 0; k < transform_sz; k++) // 16
                {
                    blendedTransforms.data<float>()[ind1 + k] += w * joint_rel_transforms.data<float>()[ind2 + k];
                }
            }
        }
    }
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void smpl::LinearBlend(cnpy::NpyArray& v_shaped, cnpy::NpyArray& weights, cnpy::NpyArray& joint_rel_transforms, cnpy::NpyArray& posed_vertices)
{
    cloneArray(v_shaped, posed_vertices);
    cnpy::NpyArray blendedTransforms;
    BlendTransforms(weights, joint_rel_transforms, blendedTransforms);

#pragma omp parallel for
    for (int i = 0; i < v_shaped.shape[0]; ++i)
    {
        float v_in[4] = { 0,0,0,1 };
        float v_out[4] = { 0,0,0,1 };
        memcpy(v_in, v_shaped.data<float>() + 3 * i, 3 * sizeof(float));
        float* T = blendedTransforms.data<float>() + 16 * i;
             
        for (int ii = 0; ii < 4; ++ii)
        {
            float x = 0;
            for (int jj = 0; jj < 4; ++jj)
            {
                x += T[ii*4+jj]*v_in[jj];
            }
            v_out[ii] = x;
        }        
        posed_vertices.data<float>()[3 * i + 0] = v_out[0];
        posed_vertices.data<float>()[3 * i + 1] = v_out[1];
        posed_vertices.data<float>()[3 * i + 2] = v_out[2];
    }
}
//-----------------------------------
// skeleton structure extraction
//-----------------------------------
void smpl::Skeleton::parse(cnpy::NpyArray& Kinematic_tree)
{
 
    int* kinematic_tree = Kinematic_tree.data<int>();
    int N_nodes = Kinematic_tree.shape[1];
    for (int i = 0; i < N_nodes; ++i)
    {
        l1.push_back(kinematic_tree[i]);
        l2.push_back(kinematic_tree[i + N_nodes]);
    }
    mas = std::vector< std::vector<int> >(N_nodes, std::vector<int>(N_nodes, 0));
    level = std::vector<int>(N_nodes, 0);
    has_child = std::vector<int>(N_nodes, 0);
    paths = std::vector < std::vector<int> >(N_nodes);

    root_ind = -1;
    for (int i = 0; i < l1.size(); ++i)
    {
        if (l1[i] == -1)
        {
            root_ind = l2[i];
        }
        else if (l2[i] == -1)
        {
            root_ind = l1[i];
        }
        else
        {
            mas[l1[i]][l2[i]] = 1;
            mas[l2[i]][l1[i]] = 1;
        }
    }

    std::cout << "Root node = " << root_ind << std::endl;

    std::vector<int> nodes(N_nodes, 0); // вершины графа

                                       //cout << "N = "; cin >> req; req--;
    for (int ind = 0; ind < N_nodes; ++ind)
    {
        paths[ind].push_back(ind);
        int req = ind;
        std::queue<int> Queue;
        std::stack<Edge> Edges;
        Edge e;
        std::fill(nodes.begin(), nodes.end(), 0);
        Queue.push(root_ind); // помещаем в очередь первую вершину
        while (!Queue.empty())
        {
            int node = Queue.front(); // извлекаем вершину
            Queue.pop();
            nodes[node] = 2; // отмечаем ее как посещенную
            for (int j = 0; j < N_nodes; j++)
            {
                if (mas[node][j] == 1 && nodes[j] == 0)
                { // если вершина смежная и не обнаружена
                    Queue.push(j); // добавляем ее в очередь
                    nodes[j] = 1; // отмечаем вершину как обнаруженную
                    e.begin = node; e.end = j;
                    Edges.push(e);
                    if (node == req) break;
                }
            }
        }

        int len = 0;
        while (!Edges.empty())
        {
            e = Edges.top();
            Edges.pop();
            if (e.end == req)
            {
                req = e.begin;
                paths[ind].push_back(req);
                has_child[req] = 1;
                len++;
            }

        }
        level[ind] = len;
    }

    std::cout << " Node levels: " << std::endl;
    for (int i = 0; i < level.size(); ++i)
    {
        std::cout << i << " : " << level[i] << std::endl;
    }
    std::cout << " ---------------------- " << std::endl;

    int max_level = *std::max_element(level.begin(), level.end());
    std::cout << " max_level=" << max_level << std::endl;

    std::vector < std::vector<int> > levels(max_level + 1);
    for (int i = 0; i < level.size(); ++i)
    {
        levels[level[i]].push_back(i);
    }

    for (int i = 0; i < max_level + 1; ++i)
    {
        std::cout << " Level #" << i << " : ";
        for (int j = 0; j < levels[i].size(); ++j)
        {
            std::cout << levels[i][j] << ",";
        }
        std::cout << std::endl;
    }


    for (int j = 0; j < paths.size(); j++)
    {
        std::reverse(paths[j].begin(), paths[j].end());
        if (has_child[j] == 0 && paths[j].size() > 0)
        {
            chains.push_back(paths[j]);
        }
    }

    for (int i = 0; i < chains.size(); i++)
    {
        std::cout << " chain #" << i << " : ";
        for (int j = 0; j < chains[i].size(); j++)
        {
            std::cout << chains[i][j] << " ";
        }
        std::cout << std::endl;
    }
}
//-----------------------------------
//
//-----------------------------------
void apply_transform(float* mat4x4, float* P, float* dst)
{
    int i, k;
    float* tmp = new float[4];
    float p_rel[4];
    p_rel[0] = P[0];
    p_rel[1] = P[1];
    p_rel[2] = P[2];
    p_rel[3] = 0;
    for (i = 0; i < 4; i++)
    {
        tmp[i] = 0;
        for (k = 0; k < 4; k++)
        {
            tmp[i] += mat4x4[i * 4 + k] * p_rel[k];
        }
    }
    memcpy(dst, tmp, 3 * sizeof(float));
    delete[] tmp;
}
//-----------------------------------
// Set skeleton pose
//-----------------------------------
void smpl::Skeleton::setPose(std::vector<float>& joints_rots_vector)
{
    this->joints_rots_vector.assign(joints_rots_vector.begin(), joints_rots_vector.end());    

    if (hands_mean.shape.size() > 0)
    {
#pragma omp parallel for    
        for (size_t i = 3; i < this->joints_rots_vector.size(); i++)
        {
            this->joints_rots_vector[i] += hands_mean.data<float>()[i - 3];
        }
    }
    // relative rotation of joints, same as joints_rots_vector, but in matrix form
    cnpy::NpyArray template_joints_relative_rotations;
    // compute rotation matrices
    posemap_axisang(this->joints_rots_vector, pose_maps, template_joints_relative_rotations);
    template_joints_rel_coords = cnpy::NpyArray(template_joints_abs_coords.shape, template_joints_abs_coords.word_size, template_joints_abs_coords.fortran_order);
    memcpy(template_joints_rel_coords.data<float>(), template_joints_abs_coords.data<float>() , 3 * sizeof(float));
    // compute relative joint offsets
    for (int i = 0; i < chains.size(); i++)
    {
#pragma omp parallel for
        for (int j = 1; j < chains[i].size(); j++)
        {
            size_t joint_ind1 = chains[i][j - 1];
            size_t joint_ind2 = chains[i][j];            
            template_joints_rel_coords.data<float>()[joint_ind2 * 3 + 0] = template_joints_abs_coords.data<float>()[joint_ind2 * 3 + 0] - template_joints_abs_coords.data<float>()[joint_ind1 * 3 + 0];
            template_joints_rel_coords.data<float>()[joint_ind2 * 3 + 1] = template_joints_abs_coords.data<float>()[joint_ind2 * 3 + 1] - template_joints_abs_coords.data<float>()[joint_ind1 * 3 + 1];
            template_joints_rel_coords.data<float>()[joint_ind2 * 3 + 2] = template_joints_abs_coords.data<float>()[joint_ind2 * 3 + 2] - template_joints_abs_coords.data<float>()[joint_ind1 * 3 + 2];
        }
    }
    memset(template_joints_rel_coords.data<float>(), 0, sizeof(float) * 3);
    // compute relative transforms of joints1
    makeTrasformsMatrix(template_joints_relative_rotations, template_joints_rel_coords, joint_rel_transforms);

    cloneArray(joint_rel_transforms, joint_abs_transforms);
    std::vector<int> posed(joint_rel_transforms.shape[0], 0);
    for (int i = 0; i < chains.size(); i++)
    {
#pragma omp parallel for
        for (int j = 1; j < chains[i].size(); j++)
        {
            size_t joint_ind1 = chains[i][j - 1];
            size_t joint_ind2 = chains[i][j];
            // now need to apply transform to joints
            float* A   = joint_rel_transforms.data<float>() + joint_ind2 * 4 * 4;
            float* B   = joint_abs_transforms.data<float>() + joint_ind1 * 4 * 4;
            float* dst = joint_abs_transforms.data<float>() + joint_ind2 * 4 * 4;
            if (!posed[joint_ind2])
            {
                posed[joint_ind2] = 1;
                M4x4_SSE(B, A, dst);
                //matMul4x4(B, A, dst);
            }
        }
    }   
    cloneArray(template_joints_abs_coords, posed_joints_rel_coords);
    for (int n = 0; n < joint_abs_transforms.shape[0]; n++)
    {
        float* T = joint_abs_transforms.data<float>() + n * 4 * 4;
        float* P = &(template_joints_abs_coords.data<float>()[n * 3]);
        float* Pt = &(posed_joints_rel_coords.data<float>()[n * 3]);
        apply_transform(T, P, Pt);
    }
    cloneArray(joint_abs_transforms, joint_rel_transforms);
#pragma omp parallel for
    for (int n = 0; n < joint_abs_transforms.shape[0]; n++)
    {
        float* dst = joint_rel_transforms.data<float>() + n * 4 * 4;
        float* P = posed_joints_rel_coords.data<float>()+n * 3;
        dst[4 * 0 + 3] -= P[0];
        dst[4 * 1 + 3] -= P[1];
        dst[4 * 2 + 3] -= P[2];
    }
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void computeBB(cnpy::NpyArray& vertices, std::vector<float>& center, std::vector<float>& size)
{
    center.resize(3);
    size.resize(3);

}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void smpl::loadModel(std::string filename)
{
    std::cout << "---------------------------" << std::endl;
    std::cout << "Loding SMPL model file : " << filename << std::endl;
    std::cout << "---------------------------" << std::endl;
    npz_map = cnpy::npz_load(filename);
    // kinematicTree
    kinematicTree = npz_map["kintree_table"]; // 2,n_joints
    std::cout << "kinematicTree" << std::endl;
    std::cout << "kinematicTree shape=" << "[" << kinematicTree.shape[0] << "," << kinematicTree.shape[1] << "]" << std::endl;
    std::cout << "-----------------" << std::endl;
    skeleton.parse(kinematicTree);
    // face indices
    faceIndices = npz_map["f"]; // n_faces,3
    face_index_num = faceIndices.shape[0];
    COUT_VAR(face_index_num);
    std::cout << "faceIndices shape ";
    printDims(faceIndices);
    std::cout << std::endl;
    // mean mesh vertices
    vertices_template = npz_map["v_template"]; // n_vert,3
    vertex_num = vertices_template.shape[0]; // n_vert,3
    COUT_VAR(vertex_num);
    std::cout << "vertices_template shape:";
    printDims(vertices_template);
    std::cout << std::endl;
    // shape basis
    shapeBlendBasis = npz_map["shapedirs"]; // n_vert,3,shape_basis_dim
    shape_basis_dim = shapeBlendBasis.shape[2];
    COUT_VAR(shape_basis_dim);
    std::cout << "shapeBlendBasis shape:";
    printDims(shapeBlendBasis);
    std::cout << std::endl;
    // pose basis
    poseBlendBasis = npz_map["posedirs"];// n_vert,3,pose_basis_dim
    pose_basis_dim = poseBlendBasis.shape[2];
    COUT_VAR(pose_basis_dim);
    std::cout << "poseBlendBasis shape:";
    printDims(poseBlendBasis);
    std::cout << std::endl;
    jointRegressor = npz_map["J_regressor"];// (n_joints, n_vert)  
    joint_num = jointRegressor.shape[0];
    COUT_VAR(joint_num);
    std::cout << "jointRegressor shape:";
    printDims(jointRegressor);
    std::cout << std::endl;
    weights = npz_map["weights"];// (n_vert,n_joints)
    std::cout << "weights shape:";
    printDims(weights);
    std::cout << std::endl;

hands_components=npz_map["hands_components"];
std::cout << "hands_components:";
printDims(hands_components);
std::cout << std::endl;

hands_coeffs = npz_map["hands_coeffs"];
std::cout << "hands_coeffs";
printDims(hands_coeffs);
std::cout << std::endl;
skeleton.hands_mean = npz_map["hands_mean"];
std::cout << "hands_meean";
printDims(skeleton.hands_mean);
std::cout << std::endl;
vetexNormals = cnpy::NpyArray(vertices_template.shape, vertices_template.word_size, vertices_template.fortran_order);
normalsComputer = new NormalsComputer(faceIndices.as_vec<unsigned int>());
    std::cout << "---------------------------" << std::endl;
    std::cout << " Successfully loaded " << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::vector<size_t> Beta_shape = { (size_t)shape_basis_dim };
    Beta= cnpy::NpyArray(Beta_shape, sizeof(float), false);
    std::vector<float> shape(shape_basis_dim,0);
    std::vector<float> pose(joint_num * 3 , 0);
    updateMesh(shape,pose);
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
void smpl::updateMesh(std::vector<float>& shape, std::vector<float>& pose)
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    assert(joint_num*3 == pose.size());
    // ----------------------------------		
     // Compute initial shape and skeleton
     // ----------------------------------
    float* beta = nullptr;
    if (shape.size() == 0)
    {
        cloneArray(vertices_template, v_posed);
    }
    else
    {
        assert(shape_basis_dim == shape.size());
        beta = Beta.data<float>();
        memcpy(beta, shape.data(), shape.size() * sizeof(float));
        computeBlendShape(Beta, vertices_template, shapeBlendBasis, v_shaped);
        computeJointsFromShape(v_shaped, jointRegressor, jointCoords);
    }
    cloneArray(jointCoords,skeleton.template_joints_abs_coords);
    //print2dArray(jointCoords);    
    // --------------
    // Compute transflormations
    // --------------
    skeleton.setPose(pose);

    computeBlendPose(skeleton.pose_maps, v_shaped, poseBlendBasis, v_posed);
    // --------------
    // Apply blend to mesh
    // --------------
    LinearBlend(v_posed, weights, skeleton.joint_rel_transforms, posed_vertices);
    normalsComputer->getVertexNormals(vertex_num * 3, posed_vertices.data<float>(), vetexNormals.data<float>());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Elapseed = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
smpl::smpl()
{
    face_index_num = 0;
    vertex_num = 0;
    shape_basis_dim = 0;
    pose_basis_dim = 0;
    joint_num = 0;
}
smpl::~smpl()
{
    delete normalsComputer;
}
//--------------------------------------------------------------------------
// Compute shape driven by Beta coefficients.
// The result is the combination of mean shape and 
// eigen shapes multiplied by shape coefficients.
//--------------------------------------------------------------------------
void smpl::computeBlendShape(cnpy::NpyArray& Beta, cnpy::NpyArray& MeanShape, cnpy::NpyArray& ShapeBlendBasis, cnpy::NpyArray& V_Shaped)
{
    float* beta = Beta.data<float>();
    float* shapeBlendBasis = ShapeBlendBasis.data<float>();
    float* meanShape = MeanShape.data<float>();
    int vertex_num = MeanShape.shape[0];
    int shapeBasisDim = ShapeBlendBasis.shape[2];
    V_Shaped = cnpy::NpyArray(MeanShape.shape, MeanShape.word_size, MeanShape.fortran_order);
    float* v_shaoed = V_Shaped.data<float>();

    for (int i = 0; i < vertex_num; ++i)
    {
        int offset = 3 * shapeBasisDim * i;
        for (int k = 0; k < 3; ++k)
        {
            int stride = shapeBasisDim * k;
            float x = meanShape[3 * i + k];
            for (int j = 0; j < shapeBasisDim; ++j)
            {
                x += shapeBlendBasis[offset + stride + j] * beta[j];
            }
            v_shaoed[3 * i + k] = x;
        }
    }
}
//--------------------------------------------------------------------------
// When we apply linear blending the shape distorted.
// The shape eigen basis tries to fix this.
//--------------------------------------------------------------------------
void smpl::computeBlendPose(cnpy::NpyArray& PoseMap, cnpy::NpyArray& V_Shaped, cnpy::NpyArray& PoseBlendBasis, cnpy::NpyArray& V_Posed)
{
    bool resultValid = false;
    if (PoseBlendBasis.shape.size() == 3)
    {
        if (PoseMap.shape[0] - 9 == PoseBlendBasis.shape[2])
        {
            resultValid = true;
            float* poseMap = PoseMap.data<float>() + 9; // skip root matrix
            float* poseBlendBasis = PoseBlendBasis.data<float>();
            float* meanShape = V_Shaped.data<float>();
            int vertex_num = V_Shaped.shape[0];
            int poseBasisDim = PoseBlendBasis.shape[2];
            V_Posed = cnpy::NpyArray(V_Shaped.shape, V_Shaped.word_size, V_Shaped.fortran_order);
            float* v_posed = V_Posed.data<float>();

            for (int i = 0; i < vertex_num; ++i)
            {
                int offset = 3 * poseBasisDim * i;
                for (int k = 0; k < 3; ++k)
                {
                    float x = meanShape[3 * i + k];                    
                    int stride = poseBasisDim * k;
                    for (int j = 0; j < poseBasisDim; ++j)
                    {
                        x += poseBlendBasis[offset + stride + j] * poseMap[j];
                    }
                    v_posed[3 * i + k] = x;
                }
            }
        }
    }
    // If model does not contains pose deformarion basis info
    if (!resultValid)
    {
        memcpy(V_Posed.data<float>(), V_Shaped.data<float>(), V_Shaped.num_bytes());
    }
}
//--------------------------------------------------------------------------
// Joint poses defined as wheighted sum of mesh vertices
//--------------------------------------------------------------------------
void smpl::computeJointsFromShape(cnpy::NpyArray& Shape, cnpy::NpyArray& JointRegressor, cnpy::NpyArray& JointCoords)
{
    float* jointRegressor = JointRegressor.data<float>();
    float* shape = Shape.data<float>();
    int vertex_num = Shape.shape[0];
    int joint_num = JointRegressor.shape[0];
    std::vector<size_t> jointsCoords_size = { (size_t)joint_num ,3 };
    JointCoords = cnpy::NpyArray(jointsCoords_size, Shape.word_size, Shape.fortran_order);
    float* jointCoords = JointCoords.data<float>();
    for (int j = 0; j < joint_num; ++j)
    {
        for (int k = 0; k < 3; ++k)
        {
            float x = 0;
            for (int i = 0; i < vertex_num; ++i)
            {
                x += shape[3 * i + k] * jointRegressor[vertex_num * j + i];
            }
            jointCoords[3 * j + k] = x;
        }
    }
}
//--------------------------------------------------------------------------
//
//--------------------------------------------------------------------------
