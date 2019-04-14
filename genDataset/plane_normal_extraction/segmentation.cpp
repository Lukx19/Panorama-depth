#include <iostream>
#define PCL_NO_PRECOMPILE
#include <math.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <vector>
#define cimg_display 0
#define cimg_use_tiff
#include "CImg.h"

struct PointXYZLUV {
  PCL_ADD_POINT4D;  // preferred way of adding a XYZ+padding
  float label;
  float u;
  float v;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // make sure our new allocators are aligned
} EIGEN_ALIGN16;  // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZLUV,  // here we assume a XYZ + "test" (as fields)
    (float, x, x)(float, y, y)(float, z, z)(float, label,
                                            label)(float, u, u)(float, v, v))

using namespace cimg_library;

double _max_distance = 0.04;
float _min_percentage = 5;
double normal_radius_search = 0.09;  // 9cm
size_t max_planes = 5;

class Color {
 private:
  uint8_t r;
  uint8_t g;
  uint8_t b;

 public:
  Color(uint8_t R, uint8_t G, uint8_t B) : r(R), g(G), b(B) {}

  void getColor(uint8_t &R, uint8_t &G, uint8_t &B) {
    R = r;
    G = g;
    B = b;
  }
  void getColor(double &rd, double &gd, double &bd) {
    rd = (double)r / 255;
    gd = (double)g / 255;
    bd = (double)b / 255;
  }
  uint32_t getColor() {
    return ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
  }
};

auto createColors(size_t n_colors) -> std::vector<Color> {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  std::vector<Color> colors;
  for (int i = 0; i < n_colors; i++) {
    while (r < 70 && g < 70 && b < 70) {
      r = rand() % (255);
      g = rand() % (255);
      b = rand() % (255);
    }
    Color c(r, g, b);
    r = 0;
    g = 0;
    b = 0;
    colors.push_back(c);
  }
  return colors;
}

void savePlanePcl(const pcl::PointCloud<PointXYZLUV>::Ptr &cloud,
                  size_t n_planes, const std::string &fname) {
  std::vector<Color> colors = createColors(n_planes);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_pub(
      new pcl::PointCloud<pcl::PointXYZRGB>);
  for (const PointXYZLUV &pt : cloud->points) {
    pcl::PointXYZRGB pt_color;
    pt_color.x = pt.x;
    pt_color.y = pt.y;
    pt_color.z = pt.z;
    size_t plane_n = static_cast<size_t>(pt.label);
    uint32_t rgb = colors.at(plane_n).getColor();
    pt_color.rgb = *reinterpret_cast<float *>(&rgb);
    cloud_pub->points.push_back(pt_color);
  }
  pcl::io::savePLYFileASCII<pcl::PointXYZRGB>(fname, *cloud_pub);
}

void savePclNormals(const pcl::PointCloud<PointXYZLUV>::Ptr &cloud,
                    const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                    const std::string &fname) {
  std::vector<Color> colors = createColors(max_planes);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_pub(
      new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    pcl::PointXYZRGBNormal pt_color;
    pt_color.x = cloud->points[i].x;
    pt_color.y = cloud->points[i].y;
    pt_color.z = cloud->points[i].z;
    pt_color.normal_x = normals->points[i].normal_x;
    pt_color.normal_y = normals->points[i].normal_y;
    pt_color.normal_z = normals->points[i].normal_z;
    pt_color.curvature = normals->points[i].curvature;
    size_t plane_n = static_cast<size_t>(cloud->points[i].label);
    uint32_t rgb = colors.at(plane_n).getColor();
    pt_color.rgb = *reinterpret_cast<float *>(&rgb);
    cloud_pub->points.push_back(pt_color);
  }
  pcl::io::savePLYFileASCII<pcl::PointXYZRGBNormal>(fname, *cloud_pub);
}

void savePlaneMasksTiff(const std::string &filename,
                        const pcl::PointCloud<PointXYZLUV>::Ptr &cloud,
                        size_t width, size_t height, size_t n_planes) {
  CImg<float> planes(width, height, 1, n_planes, 0.f);
  for (const PointXYZLUV &pt : cloud->points) {
    planes(pt.u, pt.v, 0, pt.label) = 1.f;
  }
  planes.save_tiff(filename.c_str(), 1);
}

void saveNormalsTiff(const std::string &filename,
                     const pcl::PointCloud<PointXYZLUV>::Ptr &cloud,
                     const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                     size_t width, size_t height) {
  CImg<float> normals_img(width, height, 1, 3, 0.f);
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    PointXYZLUV pt = cloud->at(i);
    pcl::Normal normal = normals->at(i);
    normals_img(pt.u, pt.v, 0, 0) = normal.normal_x;
    normals_img(pt.u, pt.v, 0, 1) = normal.normal_y;
    normals_img(pt.u, pt.v, 0, 2) = normal.normal_z;
  }
  normals_img.save_tiff(filename.c_str(), 1);
}

auto convertToPcl(const std::string &tiff_file)
    -> std::tuple<size_t, size_t, pcl::PointCloud<PointXYZLUV>::Ptr> {
  CImg<float> depth(tiff_file.c_str());
  pcl::PointCloud<PointXYZLUV>::Ptr cloud(new pcl::PointCloud<PointXYZLUV>);
  float width = depth.width();
  float height = depth.height();
  float dim = depth.depth();
  float spectrum = depth.spectrum();
  // Camera rotation angles
  float hcam_deg = 360.f;
  float vcam_deg = 180.f;
  const float PI_F = 3.14159265358979f;
  // Camera rotation angles in radians
  float hcam_rad = hcam_deg / 180.0f * PI_F;
  float vcam_rad = vcam_deg / 180.0f * PI_F;
  std::cout << "Dimensions: h: " << height << "w: " << width << "  " << dim
            << "  " << spectrum << std::endl;
  // http://mathworld.wolfram.com/SphericalCoordinates.html
  for (float v = 0; v < height; v++) {
    for (float u = 0; u < width; u++) {
      float p_theta = (u - width / 2.0f) / width * hcam_rad;
      float p_phi = -(v - height / 2.0f) / height * vcam_rad;
      // Transform into cartesian coordinates
      float radius = depth(u, v);
      // float radius = 1;
      if (radius < 0.001f || radius > 8.f) {
        continue;
      }

      float X = radius * std::cos(p_phi) * std::cos(p_theta);
      float Y = radius * std::cos(p_phi) * std::sin(p_theta);
      float Z = radius * std::sin(p_phi);
      PointXYZLUV pt;
      pt.x = X;
      pt.y = Y;
      pt.z = Z;
      pt.u = u;
      pt.v = v;
      pt.label = 0.f;
      cloud->points.push_back(std::move(pt));
    }
  }
  std::cout << "points in cloud " << cloud->size() << std::endl;
  return std::make_tuple(width, height, cloud);
}

auto estimatePlanes(const pcl::PointCloud<PointXYZLUV>::ConstPtr &cloud_in,
                    const pcl::PointCloud<pcl::Normal>::Ptr &normals)
    -> std::tuple<size_t, pcl::PointCloud<PointXYZLUV>::Ptr> {
  pcl::PointCloud<PointXYZLUV>::Ptr cloud(new pcl::PointCloud<PointXYZLUV>());
  pcl::copyPointCloud(*cloud_in, *cloud);
  std::cout << "points in cloud2 " << cloud->size() << std::endl;
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  pcl::SACSegmentationFromNormals<PointXYZLUV, pcl::Normal> seg;
  pcl::ExtractIndices<PointXYZLUV> extract;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setDistanceThreshold(_max_distance);

  // Create pointcloud to publish inliers
  pcl::PointCloud<PointXYZLUV>::Ptr cloud_res(new pcl::PointCloud<PointXYZLUV>);
  int original_size = cloud->size();
  std::cout << "original size: " << original_size;
  size_t n_planes = 0;
  while (cloud->size() > original_size * _min_percentage / 100.f &&
         n_planes < max_planes) {
    // Fit a plane
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    // Check result
    if (inliers->indices.size() == 0) break;

    for (size_t i = 0; i < inliers->indices.size(); i++) {
      PointXYZLUV new_pt = cloud->points[inliers->indices[i]];
      new_pt.label = static_cast<float>(n_planes);
      cloud_res->points.push_back(new_pt);
    }
    // sigma = sqrt(sigma/inliers->indices.size());

    // Extract inliers
    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    pcl::PointCloud<PointXYZLUV> cloudF;
    extract.filter(cloudF);
    cloud->swap(cloudF);

    // Display infor
    // "%s: mean error: %f(mm), standard deviation: %f (mm), max error:
    // %f(mm)",_name.c_str(),mean_error,sigma,max_error);
    std::cout << "poitns left in cloud " << cloud->width * cloud->height
              << std::endl;
    // Nest iteration
    n_planes++;
  }
  return std::make_tuple(n_planes, cloud_res);
}

auto estimatePlanesGrow(const pcl::PointCloud<PointXYZLUV>::ConstPtr &cloud_in,
                        const pcl::PointCloud<pcl::Normal>::Ptr &normals)
    -> std::tuple<size_t, pcl::PointCloud<PointXYZLUV>::Ptr> {
  pcl::PointCloud<PointXYZLUV>::Ptr cloud(new pcl::PointCloud<PointXYZLUV>());
  pcl::copyPointCloud(*cloud_in, *cloud);

  pcl::search::KdTree<PointXYZLUV>::Ptr tree(
      new pcl::search::KdTree<PointXYZLUV>());

  pcl::RegionGrowing<PointXYZLUV, pcl::Normal> reg;
  reg.setMinClusterSize(100);
  reg.setMaxClusterSize(1000000);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud);
  // reg.setIndices (indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold(1.0);

  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  std::cout << "Number of clusters is equal to " << clusters.size()
            << std::endl;
  std::cout << "First cluster has " << clusters[0].indices.size() << " points."
            << std::endl;

  std::vector<pcl::PointIndices>::iterator i_segment;
  int cluster_n = 1;
  for (const pcl::PointIndices &cluster : clusters) {
    std::vector<int>::iterator i_point;
    for (int index : cluster.indices) {
      cloud->points[index].label = cluster_n;
    }
    ++cluster_n;
  }
  return std::make_tuple(clusters.size() + 1, cloud);
}

auto estimateNormals(const pcl::PointCloud<PointXYZLUV>::ConstPtr &cloud_in)
    -> pcl::PointCloud<pcl::Normal>::Ptr {
  // Create the normal estimation class, and pass the input dataset to it
  pcl::NormalEstimation<PointXYZLUV, pcl::Normal> ne;
  ne.setInputCloud(cloud_in);

  // Create an empty kdtree representation, and pass it to the normal
  // estimation object. Its content will be filled inside the object, based on
  // the given input dataset (as no other search surface is given).
  pcl::search::KdTree<PointXYZLUV>::Ptr tree(
      new pcl::search::KdTree<PointXYZLUV>());
  ne.setSearchMethod(tree);

  // Output datasets
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(
      new pcl::PointCloud<pcl::Normal>);

  // Use all neighbors in a sphere of radius
  // ne.setRadiusSearch(normal_radius_search);

  // K nearest points search is much faster but less precise
  ne.setKSearch(20);

  // Compute the features
  ne.compute(*cloud_normals);
  return cloud_normals;
}

int main(int argc, char *argv[]) {
  std::string fname = "default.tiff";
  if (argc > 1) {
    fname = argv[1];
  }
  size_t start_name = fname.find("_depth_");
  if (start_name == std::string::npos) {
    std::cout << "Provided file is not a depth file: " << fname << std::endl;
    return 0;
  }
  std::string planes_fname = fname;
  std::string normals_fname = fname;
  planes_fname.replace(start_name, 7, "_planes_");
  normals_fname.replace(start_name, 7, "_normals_");

  size_t width;
  size_t height;
  size_t n_planes = 0;
  pcl::PointCloud<PointXYZLUV>::Ptr cloud;
  pcl::PointCloud<PointXYZLUV>::Ptr planes_cloud;
  pcl::PointCloud<pcl::Normal>::Ptr normals;
  std::tie(width, height, cloud) = convertToPcl(fname);
  normals = estimateNormals(cloud);
  // savePclNormals(cloud, normals, fname + "normals.ply");
  saveNormalsTiff(normals_fname, cloud, normals, width, height);
  // std::tie(n_planes, planes_cloud) = estimatePlanes(cloud, normals);
  std::tie(n_planes, planes_cloud) = estimatePlanesGrow(cloud, normals);
  // savePlanePcl(planes_cloud, n_planes, planes_fname + ".ply");
  savePlaneMasksTiff(planes_fname, planes_cloud, width, height, n_planes);
}
