#pragma once

#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>

namespace rc {

    struct Ray;
    typedef struct Hit {
        long int id = -1;
        float t = -1;
        glm::vec3 n;
    } Hit;


    /* Axis-aligned bounding box */
    typedef struct AABBHits {
        float t_begin = -1;
        float t_end = -1;
        inline bool hit() {
            return this->t_begin < this->t_end && this->t_end > 0;
        }
        inline bool inside() {
            return this->t_begin < 0 && this->t_end > 0;
        }
    } AABBHit;

    typedef struct AABB {
        glm::vec3 min;  // min corner
        glm::vec3 max;  // max corner
        AABBHit intersected_by(const struct Ray &ray) const;
        bool intersects(const AABB &aabb) const;
    } AABB;

    namespace alg {
        AABBHit ray_aabb_intersection(const struct Ray &ray, const AABB &aabb);
        bool aabb_aabb_intersection(const AABB &aabb1, const AABB &aabb2);
    };


    /* Geometry */
    enum GeometryType {
        GEOMETRY_UNKNOWN,
        GEOMETRY_TRIANGLE,
        GEOMETRY_DISK,
        GEOMETRY_SPHERE
    };

    class Geometry {
    protected:
        GeometryType _type;
    public:
        const GeometryType &type;
        long int id;
        AABB aabb;
        Geometry() : _type(GEOMETRY_UNKNOWN), type(_type), id(-1) {}
        Geometry(GeometryType __type) : _type(__type), type(_type), id(-1) {}
        Geometry(GeometryType __type, long int _id) : _type(__type), type(_type), id(_id) {}
        virtual ~Geometry() {}
    };


    /* Triangle data structures and algorithms */
    #ifndef RC_WITHOUT_TRIANGLE
    class Triangle : public Geometry {
    public:
        glm::vec3 v[3]; // vertices
        glm::vec3 n[3]; // normals

        Triangle() : Geometry(GEOMETRY_TRIANGLE) {}
        virtual ~Triangle() {}
        static AABB compute_aabb(const Triangle &triangle);
        Hit intersected_by(const struct Ray &ray) const;
    };

    namespace alg {
        Hit ray_triangle_intersection(const struct Ray &ray, const Triangle &triangle);
    };
    #endif


    /* Disk data structures and algorithms */
    #ifndef RC_WITHOUT_DISK
    class Disk : public Geometry {
    public:
        glm::vec3 p;    // center
        glm::vec3 n;    // normal
        float r;        // radius

        Disk() : Geometry(GEOMETRY_DISK) {}
        Disk(glm::vec3 _p, glm::vec3 _n, float _r) : 
            Geometry(GEOMETRY_DISK), p(_p), n(_n), r(_r) {
            this->aabb = this->compute_aabb(*this);
        }
        Disk(long int _id, glm::vec3 _p, glm::vec3 _n, float _r) : 
            Geometry(GEOMETRY_DISK, _id), p(_p), n(_n), r(_r) {
            this->aabb = this->compute_aabb(*this);
        }
        virtual ~Disk() {}
        static AABB compute_aabb(const Disk &disk);
        Hit intersected_by(const struct Ray &ray) const;
    };
    #define Disc Disk
    #define Surfel Disk

    namespace alg {
        Hit ray_disk_intersection(const struct Ray &ray, const Disk &disk);
    };
    #endif


    /* Sphere data structures and algorithms */
    #ifndef RC_WITHOUT_SPHERE
    class Sphere : public Geometry {
    public:
        glm::vec3 p;    // center
        float r;        // radius

        Sphere() : Geometry(GEOMETRY_SPHERE) {}
        Sphere(glm::vec3 _p, float _r) : 
            Geometry(GEOMETRY_SPHERE), p(_p), r(_r) {
            this->aabb = this->compute_aabb(*this);
        }
        Sphere(long int _id, glm::vec3 _p, float _r) : 
            Geometry(GEOMETRY_SPHERE, _id), p(_p), r(_r) {
            this->aabb = this->compute_aabb(*this);
        }
        virtual ~Sphere() {}
        static AABB compute_aabb(const Sphere &sphere);
        Hit intersected_by(const struct Ray &ray) const;
    };
    #define Ball Sphere

    namespace alg {
        Hit ray_sphere_intersection(const struct Ray &ray, const Sphere &sphere);
    };
    #endif

    /* Ray definition */
    typedef struct Ray {
        glm::vec3 o;    // origin
        glm::vec3 d;    // direction
        float r;        // range

        AABBHits intersects(const AABB &aabb) const;
        Hit intersects(const Geometry *geometry) const;

        #ifndef RC_WITHOUT_TRIANGLE
        Hit intersects(const Triangle &triangle) const;
        #endif
        #ifndef RC_WITHOUT_DISK
        Hit intersects(const Disk &disk) const;
        #endif
        #ifndef RC_WITHOUT_SPHERE
        Hit intersects(const Sphere &sphere) const;
        #endif
    } Ray;


    #define KDTREE_MAX_LEAF_SIZE 10
    typedef struct KDNode {
        uint8_t axis;
        float split = -1;
        Geometry **geometries = NULL;
        long int num_geometries = -1;
        struct KDNode *left = NULL, *right = NULL;

        void construct(AABB &aabb, Geometry **geometries, const long int num_geometries);
        void free();
        Hit traverse(const Ray &ray, const AABBHits &hits) const;
    } KDNode;

    typedef struct KDTree {
        AABB aabb;
        KDNode *root;

        void construct(Geometry **geometries, const long int num_geometries);
        void free();
        Hit traverse(const Ray &ray) const;
    } KDTree;



    /* Ray casting functions */
    Hit cast_ray(const Ray &ray, Geometry **geometries, const long int num_geometries);

    void cast_rays_naive(
        const Ray *rays, const long int num_rays,
        Geometry **geometries, const long int num_geometries,
        Hit *result
    );

    void cast_rays_with_kdtree(
        const Ray *rays, const long int num_rays,
        Geometry **geometries, const long int num_geometries,
        Hit *result
    );

};
