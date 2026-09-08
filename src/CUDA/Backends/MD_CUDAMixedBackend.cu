/*
 * MD_CUDAMixedBackend.cu
 *
 *  Created on: 22/feb/2013
 *      Author: lorenzo */

#include "MD_CUDAMixedBackend.h"

#include "CUDA_mixed.cuh"

#include "../../Backends/Thermostats/LangevinThermostat.h"

#include <cmath>

__global__ void setup_curand_mixed(curandState *rand_state, const llint seed, const int N) {
	if(IND >= N) return;

	curand_init(seed, IND, 0, &rand_state[IND]);
}

CUDAMixedBackend::CUDAMixedBackend() : MD_CUDABackend() {
	_d_possd = nullptr;
	_d_velsd = nullptr;
	_d_Lsd = nullptr;
	_d_orientationsd = nullptr;
}

CUDAMixedBackend::~CUDAMixedBackend(){
	if(_d_possd != nullptr) {
		CUDA_SAFE_CALL( cudaFree(_d_possd) );
		CUDA_SAFE_CALL( cudaFree(_d_orientationsd) );
		CUDA_SAFE_CALL( cudaFree(_d_velsd) );
		CUDA_SAFE_CALL( cudaFree(_d_Lsd) );
	}

	if(_d_langevin_c_rand_state != nullptr) {
		CUDA_SAFE_CALL(cudaFree(_d_langevin_c_rand_state));
	}
}

bool CUDAMixedBackend::_supports_langevin_c_integrator() const {
	return true;
}

void CUDAMixedBackend::get_settings(input_file &inp) {
	MD_CUDABackend::get_settings(inp);

	char thermostat_type[512] = "no";
	getInputString(&inp, "thermostat", thermostat_type, 0);
	_use_langevin_c = !strncmp(thermostat_type, "langevin_c", 512);
	if(_use_langevin_c) {
		LangevinThermostat langevin_params;
		langevin_params.get_settings(inp);
		langevin_params.init();

		const double gamma_trans = langevin_params.gamma_trans();
		const double gamma_rot = langevin_params.gamma_rot();

		_langevin_c_a_trans = exp(-gamma_trans * this->_dt);
		_langevin_c_b_trans = sqrt(-this->_T * expm1(-2. * gamma_trans * this->_dt));
		_langevin_c_a_rot = exp(-gamma_rot * this->_dt);
		_langevin_c_b_rot = sqrt(-this->_T * expm1(-2. * gamma_rot * this->_dt));

		OX_LOG(Logger::LOG_INFO, "Enabled CUDA mixed 'langevin_c' integrator: a_t=%g b_t=%g a_r=%g b_r=%g", _langevin_c_a_trans, _langevin_c_b_trans, _langevin_c_a_rot, _langevin_c_b_rot);
	}
}

void CUDAMixedBackend::init() {
	MD_CUDABackend::init();

	_vec_sized = ((size_t) N()) * sizeof(LR_double4);
	_orient_sized = ((size_t) N()) * sizeof(GPU_quat_double);
	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<LR_double4>(&_d_possd, _vec_sized) );
	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<LR_double4>(&_d_velsd, _vec_sized) );
	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<LR_double4>(&_d_Lsd, _vec_sized) );
	CUDA_SAFE_CALL( GpuUtils::LR_cudaMalloc<GPU_quat_double>(&_d_orientationsd, _orient_sized) );

	_float4_to_LR_double4(_d_poss, _d_possd);
	_quat_float_to_quat_double(_d_orientations, _d_orientationsd); 
	_float4_to_LR_double4(_d_vels, _d_velsd);
	_float4_to_LR_double4(_d_Ls, _d_Lsd);

	if(_use_langevin_c) {
		CUDA_SAFE_CALL(GpuUtils::LR_cudaMalloc<curandState>(&_d_langevin_c_rand_state, N() * sizeof(curandState)));
		setup_curand_mixed<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
			(_d_langevin_c_rand_state, lrand48(), N());
		CUT_CHECK_ERROR("setup_curand_mixed error");
	}
}

void CUDAMixedBackend::_init_CUDA_MD_symbols() {
	MD_CUDABackend::_init_CUDA_MD_symbols();

	float f_copy = _sqr_verlet_skin;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_sqr_verlet_skin, &f_copy, sizeof(float)) );
	f_copy = _dt;
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_dt, &f_copy, sizeof(float)) );
	int myN = N();
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(MD_N, &myN, sizeof(int)) );
}

void CUDAMixedBackend::_float4_to_LR_double4(float4 *src, LR_double4 *dest) {
	float4_to_LR_double4
		<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
		(src, dest);
	CUT_CHECK_ERROR("float4_to_LR_double4 error");
}

void CUDAMixedBackend::_LR_double4_to_float4(LR_double4 *src, float4 *dest) {
	LR_double4_to_float4
		<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
		(src, dest);
	CUT_CHECK_ERROR("LR_double4_to_float4 error");
}

void CUDAMixedBackend::_quat_float_to_quat_double(GPU_quat *src, GPU_quat_double *dest) {
	float4_to_LR_double4
		<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
		(src, dest);
	CUT_CHECK_ERROR("_quat_float_to_quat_double error");
}

void CUDAMixedBackend::_quat_double_to_quat_float(GPU_quat_double *src, GPU_quat *dest) {
	LR_double4_to_float4
		<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
		(src, dest);
	CUT_CHECK_ERROR("_quat_double_to_quat_float error");
}

void CUDAMixedBackend::_first_step() {
	if(_use_langevin_c) {
		first_step_mixed_langevin_c
			<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
			(_d_poss, _d_orientations, _d_possd, _d_orientationsd, _d_list_poss, _d_velsd, _d_Lsd, _d_forces, _d_torques,
			 _d_langevin_c_rand_state, _langevin_c_a_trans, _langevin_c_b_trans, _langevin_c_a_rot, _langevin_c_b_rot,
			 _d_are_lists_old, this->_any_rigid_body);
		CUT_CHECK_ERROR("first_step_mixed_langevin_c");
	}
	else {
		first_step_mixed
			<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
			(_d_poss, _d_orientations, _d_possd, _d_orientationsd, _d_list_poss, _d_velsd, _d_Lsd, _d_forces, _d_torques, _d_are_lists_old, this->_any_rigid_body);
	}
}

void CUDAMixedBackend::_rescale_positions(float4 new_Ls, float4 old_Ls) {
	MD_CUDABackend::_rescale_positions(new_Ls, old_Ls);
	_float4_to_LR_double4(_d_poss, _d_possd);
}

void CUDAMixedBackend::_sort_particles() {
	_LR_double4_to_float4(_d_possd, _d_poss);
	_LR_double4_to_float4(_d_velsd, _d_vels);
	_LR_double4_to_float4(_d_Lsd, _d_Ls);
	_quat_double_to_quat_float(_d_orientationsd, _d_orientations);
	MD_CUDABackend::_sort_particles();
	_quat_float_to_quat_double(_d_orientations, _d_orientationsd);
	_float4_to_LR_double4(_d_poss, _d_possd);
	_float4_to_LR_double4(_d_vels, _d_velsd);
	_float4_to_LR_double4(_d_Ls, _d_Lsd);
}

void CUDAMixedBackend::_forces_second_step() {
	this->_set_external_forces();
	_cuda_interaction->compute_forces(_cuda_lists, _d_poss, _d_orientations, _d_forces, _d_torques, _d_bonds, _d_cuda_box);

	second_step_mixed
		<<<_particles_kernel_cfg.blocks, _particles_kernel_cfg.threads_per_block>>>
		(_d_velsd, _d_Lsd, _d_forces, _d_torques, this->_any_rigid_body);
	CUT_CHECK_ERROR("second_step_mixed");
}

void CUDAMixedBackend::apply_simulation_data_changes() {
	// probably useless
	if(_d_possd != nullptr) {
		_LR_double4_to_float4(_d_possd, _d_poss);
		_quat_double_to_quat_float(_d_orientationsd, _d_orientations);
		_LR_double4_to_float4(_d_velsd, _d_vels);
		_LR_double4_to_float4(_d_Lsd, _d_Ls);
	}

	MD_CUDABackend::apply_simulation_data_changes();
}

void CUDAMixedBackend::apply_changes_to_simulation_data() {
	MD_CUDABackend::apply_changes_to_simulation_data();

	// the first time this method gets called all these arrays have not been
	// allocated yet. It's a bit of a hack but it's needed
	if(_d_possd != nullptr) {
		_float4_to_LR_double4(_d_poss, _d_possd);
		_quat_float_to_quat_double(_d_orientations, _d_orientationsd);
		_float4_to_LR_double4(_d_vels, _d_velsd);
		_float4_to_LR_double4(_d_Ls, _d_Lsd);
	}
}

void CUDAMixedBackend::_thermalize() {
	if(_use_langevin_c) {
		return;
	}

	if(_cuda_thermostat->would_activate(current_step())) {
		_LR_double4_to_float4(_d_velsd, _d_vels);
		_LR_double4_to_float4(_d_Lsd, _d_Ls);
		MD_CUDABackend::_thermalize();
		_float4_to_LR_double4(_d_vels, _d_velsd);
		_float4_to_LR_double4(_d_Ls, _d_Lsd);
	}
}

void CUDAMixedBackend::_update_stress_tensor() {
	_LR_double4_to_float4(_d_velsd, _d_vels);
	MD_CUDABackend::_update_stress_tensor();
}
