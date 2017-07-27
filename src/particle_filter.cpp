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
#include <cfloat>

#include "particle_filter.h"

using namespace std;
default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.

	normal_distribution<double> dist_x(x,std[0]);
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);
	num_particles = 100;

	for (int i =0 ; i< num_particles ; i++){
		Particle new_particle;
		new_particle.weight = 1.0f;
		new_particle.id = i;
		new_particle.x = dist_x(gen);
		new_particle.y = dist_y(gen);
		new_particle.theta = dist_theta(gen);
		particles.push_back(new_particle);
		weights.push_back(new_particle.weight);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// adding noise you may find std::normal_distribution and std::default_random_engine useful.

	normal_distribution<double> dist_x(0,std_pos[0]);
	normal_distribution<double> dist_y(0,std_pos[1]);
	normal_distribution<double> dist_theta(0,std_pos[2]);


	for (unsigned i = 0 ; i< num_particles ; i++){
		if(fabs(yaw_rate)<0.001) {

			particles[i].x = particles[i].x + velocity* delta_t * cos(particles[i].theta);
			particles[i].y = particles[i].y + velocity* delta_t * sin(particles[i].theta);
		}
		else{
			double c1 = (velocity/yaw_rate);
			particles[i].x = particles[i].x + c1*(sin(particles[i].theta + yaw_rate*delta_t)- sin(particles[i].theta));
			particles[i].y = particles[i].y + c1*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
		}
		particles[i].theta = particles[i].theta + yaw_rate * delta_t;

		particles[i].x += dist_x(gen);
		particles[i].y += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	for(unsigned i =0 ; i< observations.size() ; i++){
		double min_distance = INFINITY;
		int best_match_index = -1;
		LandmarkObs observation = observations[i];
		for (unsigned j=0; j<predicted.size() ; j++){
			LandmarkObs prediction = predicted[j];
			double x_diff = prediction.x - observation.x;
			double y_diff = prediction.y - observation.y;
			double distance = sqrt(pow(x_diff,2) + pow(y_diff,2));
			if(distance < min_distance){
				best_match_index = j;
				min_distance = distance;
			}
		}
		// keep the best match index in observation.id
		observations[i].id = best_match_index;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	for(unsigned i =0; i<num_particles; i++){
		Particle particle = particles[i];

		//-------------------------------------------------------------------
		//1- finding landmarsk that are within sensor_range
		//-------------------------------------------------------------------
		std::vector<LandmarkObs> landmarks_in_range;
		for(unsigned j=0; j <map_landmarks.landmark_list.size() ; j++){
		   LandmarkObs landmark;
		   landmark.x= map_landmarks.landmark_list[j].x_f;
			 landmark.y= map_landmarks.landmark_list[j].y_f;
			 landmark.id = map_landmarks.landmark_list[j].id_i;

			 double x_diff = landmark.x - particle.x;
			 double y_diff = landmark.y - particle.y;
			 if (sqrt(pow(x_diff,2) + pow(y_diff,2))<= sensor_range) {
				 	landmarks_in_range.push_back(landmark);
				}
		}

		//-------------------------------------------------------------------
		//2- transforming observation coordinates into particle coordinates (http://planning.cs.uiuc.edu/node99.html)
		//-------------------------------------------------------------------
		std::vector<LandmarkObs> observations_tranformed;
		for(unsigned j=0; j< observations.size(); j++){
			LandmarkObs original_observation = observations[j];
			LandmarkObs transformed_observation;
			transformed_observation.x = particle.x + original_observation.x * cos(particle.theta) - original_observation.y * sin(particle.theta);
			transformed_observation.y = particle.y + original_observation.x * sin(particle.theta) + original_observation.y * cos(particle.theta) ;
			transformed_observation.id = original_observation.id;
			observations_tranformed.push_back(std::move(transformed_observation));
		}

		//-------------------------------------------------------------------
		//3- associate predicted_landmarks to observations
		//-------------------------------------------------------------------
		dataAssociation(landmarks_in_range, observations_tranformed);

		//-------------------------------------------------------------------
		//4- update weights
		//-------------------------------------------------------------------
		double new_weight = 1.0f;
		for (unsigned j=0; j< observations_tranformed.size(); j++){

			LandmarkObs observation_transformed = observations_tranformed[j];
			LandmarkObs closest_landmark = landmarks_in_range[observation_transformed.id];

			double x_diff = observation_transformed.x - closest_landmark.x;
			double y_diff = observation_transformed.y - closest_landmark.y;
			double cov_x = pow(std_landmark[0],2);
			double cov_y = pow(std_landmark[1],2);
			double temp = (pow(x_diff,2)/cov_x) + (pow(y_diff,2)/cov_y);
			temp = (1/(2.0 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-0.5*temp);
			new_weight *=temp;
		}
		particles[i].weight = new_weight;
		weights[i] = new_weight;
	}

}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// using td::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<int> d(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;

	for(unsigned i = 0; i < num_particles; i++)
	{
		auto ind = d(gen);
		resampled_particles.push_back(std::move(particles[ind]));
	}
	particles = std::move(resampled_particles);
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
