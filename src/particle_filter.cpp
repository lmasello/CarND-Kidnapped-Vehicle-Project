/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 10;
  const double default_weight = 1.0;
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (unsigned int i=0; i<num_particles; i++) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = default_weight;
    particles.push_back(particle);
    weights.push_back(default_weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
   std::default_random_engine gen;
   normal_distribution<double> dist_x(0, std_pos[0]);
   normal_distribution<double> dist_y(0, std_pos[1]);
   normal_distribution<double> dist_theta(0, std_pos[2]);

   for (auto &particle : particles) {
     double theta = particle.theta;

     // Prediction models depending on the yaw_rate
     const double min_yaw_rate = 0.0001;
     if (fabs(yaw_rate) < min_yaw_rate) {
       particle.x += velocity * delta_t * cos(theta);
       particle.y += velocity * delta_t * sin(theta);
     }
     else {
       particle.x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
       particle.y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
       particle.theta += yaw_rate * delta_t;
     }

     // Add noise
     particle.x += dist_x(gen);
     particle.y += dist_y(gen);
     particle.theta += dist_theta(gen);
   }
}

void ParticleFilter::dataAssociation(const vector<LandmarkObs>& landmarks,
                                     LandmarkObs& observation) {
  double min_dist = std::numeric_limits<double>::max();
  int landmark_id = -1;
  for (auto landmark : landmarks) {
    double particle_dist = dist(observation.x, observation.y, landmark.x, landmark.y);

    if (particle_dist < min_dist) {
      landmark_id = landmark.id;
      min_dist = particle_dist;
    }
  }
  observation.id = landmark_id;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
   weights.clear();
   for (auto &particle: particles) {
     vector<int> associations;
     vector<double> sense_x, sense_y;
     vector<LandmarkObs> observations_in_map_coord;

     // Filter landmarks in sensor range
     vector<LandmarkObs> landmarks_in_range;
     // for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++) {
     for (auto map_landmark : map_landmarks.landmark_list) {
       // Map::single_landmark_s map_landmark = map_landmarks.landmark_list[j];
       if(
         (fabs(map_landmark.x_f - particle.x) <= sensor_range) ||
         (fabs(map_landmark.y_f - particle.y) <= sensor_range)
       )
         landmarks_in_range.push_back({map_landmark.id_i, map_landmark.x_f, map_landmark.y_f});
     }

     for (auto observation : observations) {
       LandmarkObs observation_in_map_coord;
       observation_in_map_coord.x = particle.x + cos(particle.theta) * observation.x - sin(particle.theta) * observation.y;
       observation_in_map_coord.y = particle.y + sin(particle.theta) * observation.x + cos(particle.theta) * observation.y;

       // Associate the observation to predicted landmarks to observation landmarks
       dataAssociation(landmarks_in_range, observation_in_map_coord);
       observations_in_map_coord.push_back(observation_in_map_coord);
       // Set associations, and sense_coordinates (useful for visualisation)
       associations.push_back(observation_in_map_coord.id);
       sense_x.push_back(observation_in_map_coord.x);
       sense_y.push_back(observation_in_map_coord.y);
     }

     SetAssociations(particle, associations, sense_x, sense_y);

     // Update the weights of the particle as the product of each measurement's Multivariate-Gaussian probability density.
     long double mvg_product = 1.0;
     for (auto observation: observations_in_map_coord) {
       LandmarkObs associated_landmark;
       for (unsigned int k=0; k<landmarks_in_range.size(); k++) {
         if (landmarks_in_range[k].id == observation.id)
           associated_landmark = landmarks_in_range[k];
       }
       long double mvg = multivariate_gaussian(associated_landmark.x, associated_landmark.y,
                                               std_landmark[0], std_landmark[1],
                                               observation.x, observation.y);
       mvg_product *= mvg;
     }
     particle.weight = mvg_product;
     weights.push_back(mvg_product);
   }

   // Normalize weights so that they can be used as probabilites for resampling
   long double sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0);
   for (auto &weight: weights)
     weight = weight / sum_weights * 1000;
}

void ParticleFilter::resample() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<Particle> updated_particles;

  std::discrete_distribution<int> d(weights.begin(), weights.end());
  for (unsigned int i=0; i<particles.size(); i++)
    updated_particles.push_back(particles[d(gen)]);

  particles = updated_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
