#include <boost/algorithm/algorithm.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/find.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Geometry>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>

#define cimg_display 0
#define cimg_use_tiff
#define cimg_use_opencv
#define cimg_plugin "cvMat_plugin.h"
#include "CImg.h"

#define IMG_WIDTH 1280
#define IMG_HEIGHT 1024

#define CROPPED_IMG_WIDTH 1230
#define CROPPED_IMG_HEIGHT 984
#define CROP_RADIUS 100

#define PANO_WIDTH 512
#define PANO_HEIGHT 256

#define RADIUS 2048 / CV_PI
#define LEVEL 3
#define IMAGE_NUM_PER_LEVEL 6

using namespace cimg_library;
// main function

/* compute all the panos given path_camera_info, the results will be saved in
the output_dir. results include color pano without blender, color pano with
blender, depth pano and visualize of the depth pano*/
void computePanos(const std::string &path_camera_info,
                  const std::string &output_dir);

// functional functions
void loadScanNames(const std::string path_camera_info,
                   std::vector<std::string> &names,
                   std::vector<std::vector<float>> &camera_intrinsics,
                   std::vector<std::vector<float>> &camera_poses);

void loadCameraPose(const std::vector<float> &camera_pose_vals,
                    Eigen::Affine3f &camera_pose);

void computeRect(const Eigen::Matrix3f &rot, const float radius_size,
                 const float fx, const float fy, const float cx, const float cy,
                 const int width, const int height, cv::Rect &rec);

void computePano(const std::string &scan_folder, const std::string &rot_name,
                 const std::vector<std::vector<float>> &camera_poses,
                 const std::vector<std::vector<float>> &camera_intrinsics,
                 cv::Mat &pano_color_init, cv::Mat &pano_depth_init,
                 cv::Mat &pano_color_blender);

void loadImages(const std::string &scan_folder, const std::string &rot_name,
                std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &depths);

void loadCameraPoses(
    const std::vector<std::vector<float>> &camera_poses,
    std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f>>
        &camera_poses_tmp);

void computeRects(const std::vector<Eigen::Affine3f,
                                    Eigen::aligned_allocator<Eigen::Affine3f>>
                      &camera_poses_tmp,
                  const std::vector<std::vector<float>> &camera_intrinsics,
                  std::vector<cv::Rect> &rects);

void computeInitPano(
    const std::vector<cv::Rect> &rects, const std::vector<cv::Mat> &imgs,
    const std::vector<cv::Mat> &depths,
    const std::vector<Eigen::Affine3f,
                      Eigen::aligned_allocator<Eigen::Affine3f>>
        &camera_poses_tmp,
    const std::vector<std::vector<float>> &camera_intrinsics,
    cv::Mat &pano_color_init, cv::Mat &pano_depth_init, cv::Mat &missing_mask);

void blenderPano(const std::vector<cv::Rect> &rects,
                 const std::vector<cv::Mat> &imgs,
                 const std::vector<Eigen::Affine3f,
                                   Eigen::aligned_allocator<Eigen::Affine3f>>
                     &camera_poses_tmp,
                 const std::vector<std::vector<float>> &camera_intrinsics,
                 cv::Mat &pano_color_blender);

// help functions

/*visualize depth image*/
cv::Mat visDepth(const cv::Mat &depth);
/*find the first part of the string splited by the splitter*/
std::string findFirstName(const std::string str, std::string splitter);

void loadScanNames(const std::string path_camera_info,
                   std::vector<std::string> &names,
                   std::vector<std::vector<float>> &camera_intrinsics,
                   std::vector<std::vector<float>> &camera_poses) {
  std::ifstream fin;
  fin.open(path_camera_info);
  if (!fin) {
    std::cerr << "Couldn't open " << path_camera_info << " for reading"
              << std::endl;
    return;
  }

  while (!fin.eof()) {
    std::string line;
    std::getline(fin, line);
    if (boost::algorithm::starts_with(line, "intrinsics_matrix")) {
      std::vector<std::string> strVec;
      boost::algorithm::split(strVec, line, boost::algorithm::is_any_of(" "));
      std::vector<float> intrinsics_vals;
      for (int j = 1; j < strVec.size(); j++) {
        intrinsics_vals.push_back(atof(strVec[j].c_str()));
      }
      for (int j = 0; j < 6; j++) {
        std::getline(fin, line);
        if (boost::algorithm::starts_with(line, "scan")) {
          strVec.clear();
          std::vector<float> poses_vals;
          boost::algorithm::split(strVec, line,
                                  boost::algorithm::is_any_of(" "));
          std::string depth_image_name = strVec[1];
          std::string color_image_name = strVec[2];
          std::string rot_name = findFirstName(depth_image_name, "_");
          for (int m = 3; m < 19; m++) {
            poses_vals.push_back(atof(strVec[m].c_str()));
          }
          camera_intrinsics.push_back(intrinsics_vals);
          camera_poses.push_back(poses_vals);
          names.push_back(rot_name);
        }
      }
    }
  }
  fin.close();
}

std::string findFirstName(const std::string str, std::string splitter) {
  std::size_t pos = str.find_first_of(splitter);
  return str.substr(0, pos);
}
void loadCameraPose(const std::vector<float> &camera_pose_vals,
                    Eigen::Affine3f &camera_pose) {
  camera_pose = Eigen::Affine3f::Identity();
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) {
      camera_pose(r, c) = camera_pose_vals[4 * r + c];
    }
  }
}
void computeRect(const Eigen::Matrix3f &rot, const float radius_size,
                 const float fx, const float fy, const float cx, const float cy,
                 const int width, const int height, cv::Rect &rec) {
  float minx = 999999;
  float miny = 999999;
  float maxx = -999999;
  float maxy = -999999;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      float z = 1;
      float x = (i - cx) / fx;
      float y = (j - cy) / fy;
      float x1 = x * rot(0, 0) + y * rot(0, 1) + z * rot(0, 2);
      float y1 = x * rot(1, 0) + y * rot(1, 1) + z * rot(1, 2);
      float z1 = x * rot(2, 0) + y * rot(2, 1) + z * rot(2, 2);
      float u = radius_size * atan2f(x1, z1);
      float w = y1 / sqrtf(x1 * x1 + y1 * y1 + z1 * z1);
      float v =
          radius_size * (static_cast<float>(CV_PI) - acosf(w == w ? w : 0));
      if (minx > u)
        minx = u;
      if (maxx < u)
        maxx = u;
      if (miny > v)
        miny = v;
      if (maxy < v)
        maxy = v;
    }
  }
  rec.x = minx;
  rec.y = miny;
  rec.width = maxx - minx + 1;
  rec.height = maxy - miny + 1;
}

void computePanos(const std::string &path_camera_info,
                  const std::string &output_dir) {

  boost::filesystem::path pt(path_camera_info);
  std::string scan_folder = pt.parent_path().parent_path().c_str();
  std::string scan_name = pt.stem().c_str();
  std::cout << scan_folder << "   " << scan_name << std::endl;
  std::vector<std::string> rot_names;
  std::vector<std::vector<float>> camera_intrinsics;
  std::vector<std::vector<float>> camera_poses;
  loadScanNames(path_camera_info, rot_names, camera_intrinsics, camera_poses);

  Eigen::Affine3f aff_mat_a3f = Eigen::Affine3f::Identity();
  Eigen::Matrix3f aff_mat_m3f;
  aff_mat_m3f =
      (Eigen::Matrix3f)Eigen::AngleAxisf(CV_PI, Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(0, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(0, Eigen::Vector3f::UnitX());
  aff_mat_a3f.linear() = aff_mat_m3f;
  int num_total = LEVEL * IMAGE_NUM_PER_LEVEL;

  for (int i = 0; i < rot_names.size(); i = i + num_total) {
    std::string rot_name = rot_names[i];
    std::cout << rot_name << std::endl;
    std::vector<std::vector<float>>::const_iterator camera_poses_first =
        camera_poses.begin() + i;
    std::vector<std::vector<float>>::const_iterator camera_poses_last =
        camera_poses.begin() + i + num_total;
    std::vector<std::vector<float>> camera_poses_sub(camera_poses_first,
                                                     camera_poses_last);
    std::vector<std::vector<float>>::const_iterator camera_intrinsics_first =
        camera_intrinsics.begin() + i;
    std::vector<std::vector<float>>::const_iterator camera_intrinsics_last =
        camera_intrinsics.begin() + i + num_total;
    std::vector<std::vector<float>> camera_intrinsics_sub(
        camera_intrinsics_first, camera_intrinsics_last);
    cv::Mat pano_color_init;
    cv::Mat pano_depth_init;
    cv::Mat pano_color_blender;
    computePano(scan_folder, rot_name, camera_poses_sub, camera_intrinsics_sub,
                pano_color_init, pano_depth_init, pano_color_blender);
    std::string base_name =
        output_dir + std::string("/") + scan_name + std::string("_") + rot_name;
    // cv::imwrite(output_dir + std::string("/") + scan_name + std::string("_")
    // +
    //                 rot_name + std::string("_color_init.png"),
    //             pano_color_init);
    // cv::imwrite(base_name + std::string("_depth_0_Left_Down.png"),
    //             pano_depth_init);
    cv::imwrite(base_name + std::string("_color_0.png"), pano_color_blender);
    // cv::imwrite(base_name + std::string("_depth_vis.png"),
    //             visDepth(pano_depth_init));
    cv::Mat metric_depth;
    pano_depth_init.convertTo(metric_depth, CV_32FC1);
    metric_depth = metric_depth / 4000;
    CImg<float> depth(metric_depth);
    depth.save_tiff((base_name + std::string("_depth_0.tiff")).c_str(), 1);
  }
}
void computePano(const std::string &scan_folder, const std::string &rot_name,
                 const std::vector<std::vector<float>> &camera_poses,
                 const std::vector<std::vector<float>> &camera_intrinsics,
                 cv::Mat &pano_color_init, cv::Mat &pano_depth_blender,
                 cv::Mat &pano_color_blender) {
  std::vector<cv::Mat> imgs;
  cv::Mat pano_depth_init;
  cv::Mat wrap_missing_mask;
  std::vector<cv::Mat> depths;

  loadImages(scan_folder, rot_name, imgs, depths);
  std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f>>
      camera_poses_tmp;
  loadCameraPoses(camera_poses, camera_poses_tmp);
  std::vector<cv::Rect> rects;
  computeRects(camera_poses_tmp, camera_intrinsics, rects);
  computeInitPano(rects, imgs, depths, camera_poses_tmp, camera_intrinsics,
                  pano_color_init, pano_depth_init, wrap_missing_mask);
  blenderPano(rects, imgs, camera_poses_tmp, camera_intrinsics,
              pano_color_blender);
  cv::resize(pano_color_blender, pano_color_blender, pano_color_init.size());
  pano_depth_blender = pano_depth_init;

  int border = ((pano_color_init.cols / 2) - pano_color_init.rows) / 2;
  cv::copyMakeBorder(pano_color_blender, pano_color_blender, border, border, 0,
                     0, cv::BORDER_CONSTANT, 0);
  cv::copyMakeBorder(pano_depth_blender, pano_depth_blender, border, border, 0,
                     0, cv::BORDER_CONSTANT, 0);
  cv::copyMakeBorder(wrap_missing_mask, wrap_missing_mask, border, border, 0, 0,
                     cv::BORDER_CONSTANT, 0);

  cv::Size final_size = cv::Size(PANO_WIDTH, PANO_HEIGHT);
  cv::resize(pano_color_blender, pano_color_blender, final_size, 0, 0,
             cv::INTER_AREA);
  cv::resize(pano_depth_blender, pano_depth_blender, final_size, 0, 0,
             cv::INTER_NEAREST);
  cv::resize(wrap_missing_mask, wrap_missing_mask, final_size, 0, 0,
             cv::INTER_NEAREST);
  // cv::inpaint(pano_depth_blender, wrap_missing_mask, pano_depth_blender, 10,
  // cv::INPAINT_NS);
  // pano_depth_blender = wrap_missing_mask;

  int offset = 40;
  pano_depth_blender(cv::Rect(0, 0, PANO_WIDTH, offset)) = cv::Scalar::all(0);
  pano_depth_blender(cv::Rect(0, PANO_HEIGHT - offset, PANO_WIDTH, offset)) =
      cv::Scalar::all(0);
}
void loadImages(const std::string &scan_folder, const std::string &rot_name,
                std::vector<cv::Mat> &imgs, std::vector<cv::Mat> &depths) {
  int crop_width = (IMG_WIDTH - CROPPED_IMG_WIDTH) / 2;
  int crop_height = (IMG_HEIGHT - CROPPED_IMG_HEIGHT) / 2;
  cv::Rect crop = cv::Rect(crop_width, crop_height, IMG_WIDTH - crop_width,
                           IMG_HEIGHT - crop_height);

  for (int m = 0; m < LEVEL; m++) {
    for (int n = 0; n < IMAGE_NUM_PER_LEVEL; n++) {
      std::string path_color_image =
          scan_folder + std::string("/undistorted_color_images/") + rot_name +
          std::string("_i") + std::to_string(m) + std::string("_") +
          std::to_string(n) + std::string(".jpg");
      std::string path_depth_image =
          scan_folder + std::string("/undistorted_depth_images/") + rot_name +
          std::string("_d") + std::to_string(m) + std::string("_") +
          std::to_string(n) + std::string(".png");
      cv::Mat image_color = cv::imread(path_color_image);
      cv::Mat image_depth = cv::imread(
          path_depth_image, CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
      cv::flip(image_color, image_color, +1);
      cv::flip(image_depth, image_depth, +1);
      cv::resize(image_color, image_color, cv::Size(IMG_WIDTH, IMG_HEIGHT));
      cv::resize(image_depth, image_depth, cv::Size(IMG_WIDTH, IMG_HEIGHT));
      image_color = image_color(crop);
      image_depth = image_depth(crop);
      imgs.push_back(image_color);
      depths.push_back(image_depth);
    }
  }
}
void loadCameraPoses(
    const std::vector<std::vector<float>> &camera_poses,
    std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f>>
        &camera_poses_tmp) {
  for (int m = 0; m < camera_poses.size(); m++) {
    std::vector<float> camera_pose_vals = camera_poses[m];
    Eigen::Affine3f camera_pose;
    loadCameraPose(camera_pose_vals, camera_pose);
    camera_poses_tmp.push_back(camera_pose);
  }
  Eigen::Affine3f camera_identity = camera_poses_tmp[6];
  Eigen::Affine3f camera_identity_inv = camera_identity.inverse();
  for (int m = 0; m < camera_poses.size(); m++) {
    Eigen::Affine3f camera_pose = camera_identity_inv * camera_poses_tmp[m];
    camera_poses_tmp[m] = camera_pose;
  }
}
void computeRects(const std::vector<Eigen::Affine3f,
                                    Eigen::aligned_allocator<Eigen::Affine3f>>
                      &camera_poses_tmp,
                  const std::vector<std::vector<float>> &camera_intrinsics,
                  std::vector<cv::Rect> &rects) {
  for (int m = 0; m < camera_poses_tmp.size(); m++) {
    Eigen::Affine3f camera_pose = camera_poses_tmp[m];
    std::vector<float> intrinsics_vals = camera_intrinsics[m];
    Eigen::Matrix3f pose_rot = camera_pose.linear();
    cv::Rect rect;
    computeRect(pose_rot, RADIUS, intrinsics_vals[0], intrinsics_vals[5],
                intrinsics_vals[2], intrinsics_vals[6], CROPPED_IMG_WIDTH,
                CROPPED_IMG_HEIGHT, rect);
    rects.push_back(rect);
  }
}
void computeInitPano(
    const std::vector<cv::Rect> &rects, const std::vector<cv::Mat> &imgs,
    const std::vector<cv::Mat> &depths,
    const std::vector<Eigen::Affine3f,
                      Eigen::aligned_allocator<Eigen::Affine3f>>
        &camera_poses_tmp,
    const std::vector<std::vector<float>> &camera_intrinsics,
    cv::Mat &pano_color_init, cv::Mat &pano_depth_init, cv::Mat &missing_mask) {
  float minx = 99999;
  float maxx = -99999;
  float miny = 999999;
  float maxy = -999999;
  for (int m = 0; m < rects.size(); m++) {
    cv::Rect rect = rects[m];
    if (minx > rect.x)
      minx = rect.x;
    if (maxx < rect.x + rect.width)
      maxx = rect.x + rect.width;
    if (miny > rect.y)
      miny = rect.y;
    if (maxy < rect.y + rect.height)
      maxy = rect.y + rect.height;
  }
  int width = maxx - minx;
  int height = maxy - miny;
  pano_color_init = cv::Mat::zeros(height, width, CV_8UC3);
  pano_depth_init = cv::Mat::zeros(height, width, CV_16UC1);
  missing_mask = cv::Mat(height, width, CV_16UC1);
  missing_mask = cv::Scalar(0);
  // rects.size()
  for (size_t m = 0; m < rects.size(); ++m) {

    Eigen::Affine3f camera_pose = camera_poses_tmp[m];
    std::vector<float> intrinsics_vals = camera_intrinsics[m];
    cv::Mat image_color = imgs[m];
    cv::Mat image_depth = depths[m];
    // cv::imwrite("depth_0_Left_Down_"+std::to_string(m)+".png", image_depth);
    for (int c = 0; c < image_color.cols; c++) {
      for (int r = 0; r < image_color.rows; r++) {
        float z = 1;
        float x = (c - intrinsics_vals[2]) / intrinsics_vals[0] * z;
        float y = (r - intrinsics_vals[6]) / intrinsics_vals[5] * z;
        float x1 = (x * camera_pose(0, 0) + y * camera_pose(0, 1) +
                    z * camera_pose(0, 2));
        float y1 = x * camera_pose(1, 0) + y * camera_pose(1, 1) +
                   z * camera_pose(1, 2);
        float z1 = x * camera_pose(2, 0) + y * camera_pose(2, 1) +
                   z * camera_pose(2, 2);

        float z_real = image_depth.at<ushort>(r, c);
        float z_real1 = z_real / 4000;
        float x_real1 = (c - intrinsics_vals[2]) / intrinsics_vals[0] * z_real1;
        float y_real1 = (r - intrinsics_vals[6]) / intrinsics_vals[5] * z_real1;
        // float x_real2 =
        //     -(x_real1 * camera_pose(0, 0) + y_real1 * camera_pose(0, 1) +
        //       z_real1 * camera_pose(0, 2));
        // float y_real2 = x_real1 * camera_pose(1, 0) +
        //                 y_real1 * camera_pose(1, 1) +
        //                 z_real1 * camera_pose(1, 2);
        // float z_real2 = x_real1 * camera_pose(2, 0) +
        //                 y_real1 * camera_pose(2, 1) +
        //                 z_real1 * camera_pose(2, 2);
        float z_real2 = std::sqrt(x_real1 * x_real1 + y_real1 * y_real1 +
                                  z_real1 * z_real1);

        float u = RADIUS * atan2f(x1, z1);
        float w = y1 / sqrtf(x1 * x1 + y1 * y1 + z1 * z1);
        float v = RADIUS * (static_cast<float>(CV_PI) - acosf(w == w ? w : 0));
        if ((u - minx) >= 0 && (u - minx) < pano_color_init.cols &&
            (v - miny) >= 0 && (v - miny) < pano_color_init.rows) {
          pano_color_init.at<cv::Vec3b>(v - miny, u - minx) =
              image_color.at<cv::Vec3b>(r, c);
          ushort depth = ushort(abs(z_real2 * 4000));
          if (depth > 0) {
            pano_depth_init.at<ushort>(v - miny, u - minx) = depth;
          }
        }
      }
    }
  }
  cv::Mat zero_depth;
  // cv::Mat missing_depth;
  cv::Mat black_parts;
  cv::transform(pano_color_init, black_parts, cv::Matx13f(1, 1, 1));
  cv::threshold(black_parts, zero_depth, 0, 255, cv::THRESH_BINARY_INV);
  // int missing_poles = 50;
  // for (int i = 0; i < missing_poles; i++)
  //   zero_depth.row(i).setTo(0);
  // for (int i = zero_depth.rows - missing_poles; i < zero_depth.rows; i++)
  //   zero_depth.row(i).setTo(0);
  zero_depth.convertTo(missing_mask, CV_8UC1, 1, 0);
  // cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3),
  // cv::Point(-1, -1));
  // cv::dilate(zero_depth, missing_depth, element, cv::Point(-1, -1), 10);
  // cv::substract(cv::Scalar(255),missing_depth,missing_depth);
  // cv::substract(cv::Scalar(255),zero_depth,zero_depth);

  // missing_mask =
  // (cv::Scalar(255) - zero_depth) - (cv::Scalar(255) - missing_depth);
}
void blenderPano(const std::vector<cv::Rect> &rects,
                 const std::vector<cv::Mat> &imgs,
                 const std::vector<Eigen::Affine3f,
                                   Eigen::aligned_allocator<Eigen::Affine3f>>
                     &camera_poses_tmp,
                 const std::vector<std::vector<float>> &camera_intrinsics,
                 cv::Mat &pano_color_blender) {

  cv::Ptr<cv::detail::Blender> blender_ =
      cv::makePtr<cv::detail::MultiBandBlender>(false);

  cv::Ptr<cv::WarperCreator> warper_ = cv::makePtr<cv::SphericalWarper>();

  cv::Ptr<cv::detail::ExposureCompensator> exposure_comp_ =
      cv::makePtr<cv::detail::BlocksGainCompensator>();

  cv::Ptr<cv::detail::SeamFinder> seam_finder_ =
      cv::makePtr<cv::detail::GraphCutSeamFinder>(
          cv::detail::GraphCutSeamFinderBase::COST_COLOR);
  float seam_scale = 0.2;

  cv::Ptr<cv::detail::RotationWarper> w =
      warper_->create(float(RADIUS * seam_scale));
  std::vector<cv::Point> corners(rects.size());
  std::vector<cv::Size> sizes(rects.size());
  std::vector<cv::UMat> masks_warped(rects.size());
  std::vector<cv::UMat> images_warped(rects.size());

  for (int m = 0; m < rects.size(); m++) {
    std::vector<float> intrinsics_vals = camera_intrinsics[m];
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = intrinsics_vals[0] * seam_scale;
    K.at<float>(0, 2) = intrinsics_vals[2] * seam_scale;
    K.at<float>(1, 1) = intrinsics_vals[5] * seam_scale;
    K.at<float>(1, 2) = intrinsics_vals[6] * seam_scale;
    Eigen::Affine3f camera_pose = camera_poses_tmp[m];
    Eigen::Matrix3f pose = camera_pose.linear();
    cv::Mat R;
    cv::eigen2cv(pose, R);
    cv::UMat img1 = imgs[m].getUMat(cv::ACCESS_READ);
    cv::UMat img;
    cv::resize(img1, img, cv::Size(), seam_scale, seam_scale);
    corners[m] = w->warp(img, K, R, cv::INTER_LINEAR, cv::BORDER_REFLECT,
                         images_warped[m]);
    sizes[m] = images_warped[m].size();
    cv::UMat mask;
    mask.create(img.size(), CV_8U);
    mask.setTo(cv::Scalar::all(255));
    w->warp(mask, K, R, cv::INTER_NEAREST, cv::BORDER_CONSTANT,
            masks_warped[m]);
  }

  exposure_comp_->feed(corners, images_warped, masks_warped);
  std::vector<cv::UMat> images_warped_f(masks_warped.size());

  for (size_t m = 0; m < masks_warped.size(); ++m)
    images_warped[m].convertTo(images_warped_f[m], CV_32F);

  seam_finder_->find(images_warped_f, corners, masks_warped);
  cv::Ptr<cv::detail::RotationWarper> w2 = warper_->create(float(RADIUS));

  std::vector<cv::Point> corners2(rects.size());
  std::vector<cv::Size> sizes2(rects.size());
  std::vector<cv::UMat> masks_warped2(rects.size());
  std::vector<cv::UMat> images_warped2(rects.size());

  for (int m = 0; m < rects.size(); m++) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    std::vector<float> intrinsics_vals = camera_intrinsics[m];
    K.at<float>(0, 0) = intrinsics_vals[0];
    K.at<float>(0, 2) = intrinsics_vals[2];
    K.at<float>(1, 1) = intrinsics_vals[5];
    K.at<float>(1, 2) = intrinsics_vals[6];
    Eigen::Affine3f camera_pose = camera_poses_tmp[m];
    Eigen::Matrix3f pose = camera_pose.linear();
    cv::Mat R;
    cv::eigen2cv(pose, R);
    cv::UMat img = imgs[m].getUMat(cv::ACCESS_READ);
    corners2[m] = w2->warp(img, K, R, cv::INTER_LINEAR, cv::BORDER_REFLECT,
                           images_warped2[m]);
    sizes2[m] = images_warped2[m].size();
    cv::UMat mask;
    mask.create(img.size(), CV_8U);
    mask.setTo(cv::Scalar::all(255));
    w2->warp(mask, K, R, cv::INTER_NEAREST, cv::BORDER_CONSTANT,
             masks_warped2[m]);
  }
  bool is_blender_prepared = false;
  cv::UMat img_warped;
  cv::UMat dilated_mask, seam_mask, mask_warped;
  for (int m = 0; m < images_warped2.size(); m++) {
    img_warped = images_warped2[m];
    mask_warped = masks_warped2[m];
    dilate(masks_warped[m], dilated_mask, cv::Mat());
    resize(dilated_mask, seam_mask, mask_warped.size());
    bitwise_and(seam_mask, mask_warped, mask_warped);
    if (!is_blender_prepared) {
      blender_->prepare(corners2, sizes2);
      is_blender_prepared = true;
    }
    exposure_comp_->apply(m, corners2[m], img_warped, mask_warped);
    cv::UMat img_warped_s;
    img_warped.convertTo(img_warped_s, CV_16S);
    blender_->feed(img_warped_s, mask_warped, corners2[m]);
  }
  cv::UMat result, result_mask;
  blender_->blend(result, result_mask);
  result.convertTo(pano_color_blender, CV_8U);
}

cv::Mat visDepth(const cv::Mat &depth) {
  double min;
  double max;
  cv::minMaxIdx(depth, &min, &max);
  cv::Mat depth_map;
  float scale = 255 / (max - min);
  depth.convertTo(depth_map, CV_8UC1, scale, -min * scale);
  cv::Mat depth_vis;
  applyColorMap(depth_map, depth_vis, cv::COLORMAP_AUTUMN);
  return depth_vis;
}

int main(int argc, char **argv) {
  cv::setNumThreads(2 * cv::getNumThreads() / 3);
  std::vector<std::string> args(argv, argv + argc);
  std::string camera_config_path = args[1];
  std::string output_dir = args[2];
  computePanos(camera_config_path, output_dir);
  return 0;
}