/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#define NUMBER_OF_PARTICLES 100

using namespace std;

static std::default_random_engine generator;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles.
	num_particles = NUMBER_OF_PARTICLES;

	// define the distribution from which particles will be generated
	std::normal_distribution<double> x_generator(x, std[0]);
	std::normal_distribution<double> y_generator(y, std[1]);
	std::normal_distribution<double> theta_generator(theta, std[2]);

	// Initialize all particles to first position (based on estimates of
	// x, y, theta and their uncertainties from GPS)
	for (int i = 0; i < num_particles; i++)
	{
		Particle p;
		p.id = particle_id_counter++;
		p.x = x_generator(generator);
		p.y = y_generator(generator);
		p.theta = theta_generator(generator);
		p.weight = 1;

		particles.push_back(p);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// define the distribution from which noises will be generated
	std::normal_distribution<double> x_noise(0, std_pos[0]);
	std::normal_distribution<double> y_noise(0, std_pos[1]);
	std::normal_distribution<double> theta_noise(0, std_pos[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];

		// bicycle motion model
		if (abs(yaw_rate) < 0.001) {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		} else {
			p.x += velocity * (sin(p.theta + delta_t * yaw_rate) - sin(p.theta)) / yaw_rate;
			p.y += velocity * (cos(p.theta) - cos(p.theta + delta_t * yaw_rate)) / yaw_rate;
			p.theta += delta_t * yaw_rate;
		}

		// add noise
		p.x += x_noise(generator);
		p.y += y_noise(generator);
		p.theta += theta_noise(generator);

		// save updated coords back
		particles[i] = p;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++) {
		double min_dist = numeric_limits<double>::max();
		for (int j = 0; j < predicted.size(); j++) {
			double dist = sqrt(pow(observations[i].x - predicted[j].x, 2) + \
			                   pow(observations[i].y - predicted[j].y, 2));
			if (dist < min_dist) {
				min_dist = dist;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.htmlf

	// calculate normalization term
	double gauss_norm = 1/(2 * M_PI * std_landmark[0] * std_landmark[1]);

	for (int i = 0; i < num_particles; i++) {
		Particle p = particles[i];
		std::vector<LandmarkObs> observations_in_map_coords;

		// transform measurement into world coordinates
		for (auto it = observations.begin(); it != observations.end(); it++) {
			LandmarkObs obs_in_map_coords;
			obs_in_map_coords.x = p.x + (cos(p.theta) * it->x) - (sin(p.theta) * it->y);
			obs_in_map_coords.y = p.y + (sin(p.theta) * it->x) + (cos(p.theta) * it->y);

			observations_in_map_coords.push_back(obs_in_map_coords);
		}

		// list of possible landmarks based on sensor range
		std::vector<LandmarkObs> predictions;
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			auto landmark = map_landmarks.landmark_list[j];
			double dist = sqrt(pow(p.x - landmark.x_f, 2) + \
			                   pow(p.y - landmark.y_f, 2));
			if (dist < sensor_range) {
				LandmarkObs prediction;
				prediction.x = landmark.x_f;
				prediction.y = landmark.y_f;
				prediction.id = landmark.id_i;
				predictions.push_back(prediction);
			}
		}

		// associate observations with potential landmarkings
		dataAssociation(predictions, observations_in_map_coords);

		// calculate weight using normalization terms and exponent
		p.weight = 1;
		p.associations.clear();
		p.sense_x.clear();
		p.sense_y.clear();
		for (int j = 0; j < observations_in_map_coords.size(); j++) {
			auto obs = observations_in_map_coords[j];
			auto feature = map_landmarks.landmark_list[obs.id - 1];
			p.associations.push_back(obs.id);
			p.sense_x.push_back(obs.x);
			p.sense_y.push_back(obs.y);

			// calculate exponent
			double exponent= (pow((obs.x - feature.x_f)/std_landmark[0], 2) + \
			                  pow((obs.y - feature.y_f)/std_landmark[1], 2))/2;

			// update weight
			p.weight *= gauss_norm * exp(-exponent);
		}

		// save particle back to vector
		particles[i] = p;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// construct a vector of weights
	vector<double> weights;
	for (auto it = particles.begin(); it != particles.end(); it++) {
		weights.push_back(it->weight);
	}

	discrete_distribution<uint32_t> d(weights.begin(), weights.end());

	vector<Particle> resampled_particles;
	for (int i = 0; i < num_particles; i++) {
		int sample_id = d(generator);
		resampled_particles.push_back(particles[sample_id]);
	}

	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
									 const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations= associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
