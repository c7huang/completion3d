#include <Python.h>
#include "ray_casting.hpp"

namespace rc {

    bool AABB::intersects(const AABB &aabb) const {
        return alg::aabb_aabb_intersection(*this, aabb);
    }
    AABBHits AABB::intersected_by(const Ray &ray) const {
        return alg::ray_aabb_intersection(ray, *this);
    }
    AABBHits Ray::intersects(const AABB &aabb) const {
        return alg::ray_aabb_intersection(*this, aabb);
    }
    namespace alg {
        AABBHits ray_aabb_intersection(const Ray &ray, const AABB &aabb) {
            AABBHits hits = {-1, -1};
            if (ray.d.x == 0 && (ray.o.x < aabb.min.x || ray.o.x > aabb.max.x))
                return hits;

            float t1, t2, tmp;

            /* X dimension */
            t1 = (aabb.min.x - ray.o.x) / ray.d.x;
            t2 = (aabb.max.x - ray.o.x) / ray.d.x;
            if (t1 > t2) {
                tmp = t1; t1 = t2; t2 = tmp;
            }
            hits.t_begin = t1;
            hits.t_end = t2;

            /* Y dimension */
            t1 = (aabb.min.y - ray.o.y) / ray.d.y;
            t2 = (aabb.max.y - ray.o.y) / ray.d.y;
            if (t1 > t2) {
                tmp = t1; t1 = t2; t2 = tmp;
            }
            if (t1 > hits.t_begin) hits.t_begin = t1;
            if (t2 < hits.t_end) hits.t_end = t2;

            /* Z dimension */
            t1 = (aabb.min.z - ray.o.z) / ray.d.z;
            t2 = (aabb.max.z - ray.o.z) / ray.d.z;
            if (t1 > t2) {
                tmp = t1; t1 = t2; t2 = tmp;
            }
            if (t1 > hits.t_begin) hits.t_begin = t1;
            if (t2 < hits.t_end) hits.t_end = t2;

            return hits;
        }

        bool aabb_aabb_intersection(const AABB &a, const AABB &b) {
            return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
                (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
                (a.min.z <= b.max.z && a.max.z >= b.min.z);
        }
    }


    #ifndef RC_WITHOUT_TRIANGLE
    AABB Triangle::compute_aabb(const Triangle &triangle) {
        // TODO: implement me
        return AABB();
    }
    Hit Triangle::intersected_by(const Ray &ray) const {
        return alg::ray_triangle_intersection(ray, *this);
    }
    Hit Ray::intersects(const Triangle &triangle) const {
        return alg::ray_triangle_intersection(*this, triangle);
    }
    Hit alg::ray_triangle_intersection(const Ray &ray, const Triangle &triangle) {
        // TODO: implement me
        return Hit();
    }
    #endif


    #ifndef RC_WITHOUT_DISK
    AABB Disk::compute_aabb(const Disk &disk) {
        glm::vec3 r = glm::vec3(disk.r, disk.r, disk.r);
        return AABB({disk.p - r, disk.p + r});
    }
    Hit Disk::intersected_by(const Ray &ray) const {
        return alg::ray_disk_intersection(ray, *this);
    }
    Hit Ray::intersects(const Disk &disk) const {
        return alg::ray_disk_intersection(*this, disk);
    }
    Hit alg::ray_disk_intersection(const Ray &ray, const Disk &disk) {
        Hit hit = {disk.id, -1, disk.n};

        float denom = glm::dot(ray.d, disk.n); 
        if (abs(denom) < 1e-6) {
            hit.t = -1;
            return hit;
        }
        hit.t = glm::dot(disk.p-ray.o, disk.n) / denom; 

        /* Check if the ray is out-of-range */
        if (ray.r > 0 && hit.t > ray.r) {
            hit.t = -1;
            return hit;
        }

        /* Check if the hit is within the radius of the disk */
        if (glm::length(ray.o+hit.t*ray.d-disk.p) > disk.r) {
            hit.t = -1;
            return hit;
        }

        return hit;
    }
    #endif


    #ifndef RC_WITHOUT_SPHERE
    AABB Sphere::compute_aabb(const Sphere &sphere) {
        glm::vec3 r = glm::vec3(sphere.r, sphere.r, sphere.r);
        return AABB({sphere.p - r, sphere.p + r});
    }
    Hit Sphere::intersected_by(const Ray &ray) const {
        return alg::ray_sphere_intersection(ray, *this);
    }
    Hit Ray::intersects(const Sphere &sphere) const {
        return alg::ray_sphere_intersection(*this, sphere);
    }
    Hit alg::ray_sphere_intersection(const Ray &ray, const Sphere &sphere) {
        // TODO: implement me
        return Hit();
    }
    #endif

    Hit Ray::intersects(const Geometry *geometry) const {
        Hit hit;
        switch(geometry->type) {
            case GEOMETRY_UNKNOWN:
                break;
            #ifndef RC_WITHOUT_TRIANGLE
            case GEOMETRY_TRIANGLE:
                hit = this->intersects(*dynamic_cast<const Triangle *>(geometry));
                break;
            #endif
            #ifndef RC_WITHOUT_DISK
            case GEOMETRY_DISK:
                hit = this->intersects(*dynamic_cast<const Disk *>(geometry));
                break;
            #endif
            #ifndef RC_WITHOUT_SPHERE
            case GEOMETRY_SPHERE:
                hit = this->intersects(*dynamic_cast<const Sphere *>(geometry));
                break;
            #endif
        }
        return hit;
    }


    void KDNode::construct(AABB &aabb, Geometry **geometries, const long int num_geometries) {
        /* Check if satisfies leaf node */
        if (num_geometries < KDTREE_MAX_LEAF_SIZE) {
            this->num_geometries = num_geometries;
            this->geometries = new Geometry *[num_geometries];
            for (long int i = 0; i < num_geometries; ++i)
                this->geometries[i] = geometries[i];
            return;
        }

        glm::vec3 diff = aabb.max - aabb.min;
        uint8_t axis[3];
        if (diff.x >= diff.y && diff.y >= diff.z) {
            axis[0] = 0; axis[1] = 1; axis[2] = 2;
        } else if (diff.x >= diff.z && diff.z >= diff.y) {
            axis[0] = 0; axis[1] = 2; axis[2] = 1;
        } else if (diff.y >= diff.x && diff.x >= diff.z) {
            axis[0] = 1; axis[1] = 0; axis[2] = 2;
        } else if (diff.y >= diff.z && diff.z >= diff.x) {
            axis[0] = 1; axis[1] = 2; axis[2] = 0;
        } else if (diff.z >= diff.x && diff.x >= diff.y) {
            axis[0] = 2; axis[1] = 0; axis[2] = 1;
        } else {
            axis[0] = 2; axis[1] = 1; axis[2] = 0;
        }

        for (int axis_i = 0; axis_i < 3; ++axis_i) {
            this->axis = axis[axis_i];

            std::vector<float> axis_values(num_geometries);
            for (long int i = 0; i < num_geometries; ++i) {
                axis_values[i] = (
                    geometries[i]->aabb.min[this->axis] + 
                    geometries[i]->aabb.max[this->axis]
                ) / 2;
            }
            std::sort(axis_values.begin(), axis_values.end());
            this->split = (axis_values[(int)(num_geometries/2)] + axis_values[(int)(num_geometries/2)+1]) / 2.0;

            AABB aabb_left = { aabb.min, aabb.max };
            aabb_left.max[this->axis] = this->split;
            AABB aabb_right = { aabb.min, aabb.max };
            aabb_right.min[this->axis] = this->split;

            /* Count number of geometries per side */
            long int num_geometries_left = 0, *indices_left = new long int[num_geometries], 
                num_geometries_right = 0, *indices_right = new long int[num_geometries];
            for (long int i = 0; i < num_geometries; ++i) {
                if (geometries[i]->aabb.intersects(aabb_left))
                    indices_left[num_geometries_left++] = i;
                if (geometries[i]->aabb.intersects(aabb_right))
                    indices_right[num_geometries_right++] = i;
            }

            /* If too many elements are shared between two nodes, 
               stop to prevent infinite loops. */
            if (num_geometries_left + num_geometries_right - num_geometries >= 0.5 * (float) num_geometries) {
                delete indices_left;
                delete indices_right;

                if (axis_i != 2) {
                    continue;
                } else {
                    this->num_geometries = num_geometries;
                    this->geometries = new Geometry *[num_geometries];
                    for (long int i = 0; i < num_geometries; ++i)
                        this->geometries[i] = geometries[i];
                    break;
                }
            }

            /* Split array and construct new ones */
            if (num_geometries_left > 0) {
                Geometry **geometries_left = new Geometry *[num_geometries_left];
                for (long int i = 0; i < num_geometries_left; ++i)
                    geometries_left[i] = geometries[indices_left[i]];
                this->left = new KDNode();
                this->left->construct(aabb_left, geometries_left, num_geometries_left);
                delete geometries_left;
            }

            if (num_geometries_right > 0) {
                Geometry **geometries_right = new Geometry *[num_geometries_right];
                for (long int i = 0; i < num_geometries_right; ++i)
                    geometries_right[i] = geometries[indices_right[i]];
                this->right = new KDNode();
                this->right->construct(aabb_right, geometries_right, num_geometries_right);
                delete geometries_right;
            }

            delete indices_left;
            delete indices_right;
            break;
        }
    }

    void KDNode::free() {
        if (this->geometries != NULL)  {
            delete this->geometries;
            this->geometries = NULL;
        }
        if (this->left != NULL) {
            this->left->free();
            delete this->left;
            this->left = NULL;
        }
        if (this->right != NULL) {
            this->right->free();
            delete this->right;
            this->right = NULL;
        }
    }

    #define KDNODE_TRAVERSE_LEFT(ray, hits) { \
        if (this->left != NULL) return this->left->traverse(ray, hits); \
        else return Hit(); }

    #define KDNODE_TRAVERSE_RIGHT(ray, hits) { \
        if (this->right != NULL) return this->right->traverse(ray, hits); \
        else return Hit(); }

    Hit KDNode::traverse(const Ray &ray, const AABBHits &hits) const {
        if (this->num_geometries >= 0) {
            Hit hit = cast_ray(ray, this->geometries, this->num_geometries);
            /* Check if the hit is inside the current node */
            if (hit.t < hits.t_begin || hit.t > hits.t_end) 
                return Hit();
            else
                return hit;
        }

        /* Ray parallel to split plane */
        if (ray.d[this->axis] == 0) {
            if (ray.o[this->axis] <= this->split)
                KDNODE_TRAVERSE_LEFT(ray, hits)
            else
                KDNODE_TRAVERSE_RIGHT(ray, hits)
        }

        /* Find intersection with the split plane */
        float t_split = (this->split - ray.o[this->axis]) / ray.d[this->axis];
        if (t_split <= hits.t_begin) { // Intersects after split plane
            if (ray.d[this->axis] > 0)
                KDNODE_TRAVERSE_RIGHT(ray, hits)
            else
                KDNODE_TRAVERSE_LEFT(ray, hits)
        } else if (t_split >= hits.t_end) { // Intersects before split plane
            if (ray.d[this->axis] > 0)
                KDNODE_TRAVERSE_LEFT(ray, hits)
            else
                KDNODE_TRAVERSE_RIGHT(ray, hits)
        } else {    // Intersects both
            if (ray.d[this->axis] > 0) {
                if (this->left != NULL) {
                    Hit hit = this->left->traverse(ray, {hits.t_begin, t_split});
                    if (hit.t > 0 && hit.t <= t_split) return hit;
                }
                return this->right->traverse(ray, {t_split, hits.t_end});
            } else {
                if (this->right != NULL) {
                    Hit hit = this->right->traverse(ray, {hits.t_begin, t_split});
                    if (hit.t > 0 && hit.t <= t_split) return hit;
                }
                return this->left->traverse(ray, {t_split, hits.t_end});
            }
        }
    }

    void KDTree::construct(Geometry **geometries, const long int num_geometries) {
        /* Compute root node aabb */
        this->aabb.min.x = this->aabb.min.y = this->aabb.min.z = std::numeric_limits<float>::max();
        this->aabb.max.x = this->aabb.max.y = this->aabb.max.z = std::numeric_limits<float>::min();
        for (long int i = 0; i < num_geometries; ++i) {
            if (geometries[i]->aabb.min.x < this->aabb.min.x)
                this->aabb.min.x = geometries[i]->aabb.min.x;
            if (geometries[i]->aabb.min.y < this->aabb.min.y)
                this->aabb.min.y = geometries[i]->aabb.min.y;
            if (geometries[i]->aabb.min.z < this->aabb.min.z)
                this->aabb.min.z = geometries[i]->aabb.min.z;
            if (geometries[i]->aabb.max.x > this->aabb.max.x)
                this->aabb.max.x = geometries[i]->aabb.max.x;
            if (geometries[i]->aabb.max.y > this->aabb.max.y)
                this->aabb.max.y = geometries[i]->aabb.max.y;
            if (geometries[i]->aabb.max.z > this->aabb.max.z)
                this->aabb.max.z = geometries[i]->aabb.max.z;
        }

        this->root = new KDNode();
        this->root->construct(this->aabb, geometries, num_geometries);
    }

    void KDTree::free() {
        this->root->free();
        delete this->root;
        this->root = NULL;
    }

    Hit KDTree::traverse(const Ray &ray) const {
        if (this->root == NULL)
            return Hit();
        
        AABBHits hits = ray.intersects(this->aabb);
        if (hits.hit()) {
            return this->root->traverse(ray, hits);
        } else {
            return Hit();
        }
    }


    Hit cast_ray(const Ray &ray, Geometry **geometries, const long int num_geometries) {
        Hit closest_hit = {-1, std::numeric_limits<float>::max(), glm::vec3()};

        for (long int gid = 0; gid < num_geometries; ++gid) {
            Hit hit = ray.intersects(geometries[gid]);
            if (hit.t > 0 && hit.t < closest_hit.t)
                closest_hit = hit;
        }
        if (closest_hit.t == std::numeric_limits<float>::max() || closest_hit.t < 0)
            closest_hit = Hit();
        return closest_hit;
    }

    void cast_rays_naive(
        const Ray *rays, const long int num_rays,
        Geometry **geometries, const long int num_geometries,
        Hit *result
    ) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (long int rid = 0; rid < num_rays; ++rid)
            result[rid] = cast_ray(rays[rid], geometries, num_geometries);
    }

    void cast_rays_with_kdtree(
        const Ray *rays, const long int num_rays,
        Geometry **geometries, const long int num_geometries,
        Hit *result
    ) {
        /* Build KDTree */
        KDTree kdtree;
        kdtree.construct(geometries, num_geometries);

        /* Search KDTree */
        #pragma omp parallel for schedule(dynamic, 1)
        for (long int rid = 0; rid < num_rays; ++rid)
            result[rid] = kdtree.traverse(rays[rid]);
        
        kdtree.free();
    }

}
