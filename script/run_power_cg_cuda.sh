#!/bin/bash
MTX_HOME="/qfs/projects/eracle/data/"
RES_OUT_HOME="./output_results"
RES_HOME="./power_results"
SCRIPT="../../lap_cg " ## This is were the command go
PRECOND=('GS_std' 'GS_it' 'GS_it2' 'it_jacobi' 'line_jacobi' 'ichol')
#MATRIX=('delaunay_n24/delaunay_n24.mtx', 'Fault_639/Fault_639.mtx', 'G3_circuit/G3_circuit.mtx', 'Serena/Serena.mtx', 't60k/t60k.mtx', 'thermal2/thermal2.mtx', 'af_0_k101/af_0_k101.mtx', 'hugebubbles-00000/hugebubbles-00000.mtx', 'adaptive/adaptive.mtx', Hook_1498/Hook_1498.mtx)
MATRIX=('Fault_639/Fault_639.mtx' 'G3_circuit/G3_circuit.mtx' 'Serena/Serena.mtx' 'thermal2/thermal2.mtx' 'af_0_k101/af_0_k101.mtx' 'Hook_1498/Hook_1498.mtx')
#MATRIX=('Fault_639/Fault_639.mtx')
RHS=('' '' '' 'thermal2/thermal2_b.mtx' 'af_0_k101/af_0_k101_b.mtx' '')

declare -A ARGX

ARGX+=(["${MATRIX[0]}_${PRECOND[0]}"]='3 3') 
ARGX+=(["${MATRIX[0]}_${PRECOND[1]}"]='-1') 
ARGX+=(["${MATRIX[0]}_${PRECOND[2]}"]='-1') 
ARGX+=(["${MATRIX[0]}_${PRECOND[3]}"]='-1') 
ARGX+=(["${MATRIX[0]}_${PRECOND[4]}"]='10 1') 
ARGX+=(["${MATRIX[0]}_${PRECOND[5]}"]='-1') 

ARGX+=(["${MATRIX[1]}_${PRECOND[0]}"]='1 1') 
ARGX+=(["${MATRIX[1]}_${PRECOND[1]}"]='-1') 
ARGX+=(["${MATRIX[1]}_${PRECOND[2]}"]='1 1') 
ARGX+=(["${MATRIX[1]}_${PRECOND[3]}"]='1 1') 
ARGX+=(["${MATRIX[1]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[1]}_${PRECOND[5]}"]='1 1') 

ARGX+=(["${MATRIX[2]}_${PRECOND[0]}"]='1 1') 
ARGX+=(["${MATRIX[2]}_${PRECOND[1]}"]='-1')
ARGX+=(["${MATRIX[2]}_${PRECOND[2]}"]='3 15') 
ARGX+=(["${MATRIX[2]}_${PRECOND[3]}"]='-1')
ARGX+=(["${MATRIX[2]}_${PRECOND[4]}"]='1 1') 
ARGX+=(["${MATRIX[2]}_${PRECOND[5]}"]='-1') 

ARGX+=(["${MATRIX[3]}_${PRECOND[0]}"]='1 2')
ARGX+=(["${MATRIX[3]}_${PRECOND[1]}"]='12 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[2]}"]='1 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[3]}"]='1 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[3]}_${PRECOND[5]}"]='1 1')

ARGX+=(["${MATRIX[4]}_${PRECOND[0]}"]='6 3')
ARGX+=(["${MATRIX[4]}_${PRECOND[1]}"]='-1')
ARGX+=(["${MATRIX[4]}_${PRECOND[2]}"]='1 3')
ARGX+=(["${MATRIX[4]}_${PRECOND[3]}"]='6 3')
ARGX+=(["${MATRIX[4]}_${PRECOND[4]}"]='-1')
ARGX+=(["${MATRIX[4]}_${PRECOND[5]}"]='1 1')

ARGX+=(["${MATRIX[5]}_${PRECOND[0]}"]='5 5')
ARGX+=(["${MATRIX[5]}_${PRECOND[1]}"]='-1')
ARGX+=(["${MATRIX[5]}_${PRECOND[2]}"]='25 25')
ARGX+=(["${MATRIX[5]}_${PRECOND[3]}"]='-1')
ARGX+=(["${MATRIX[5]}_${PRECOND[4]}"]='1 1')
ARGX+=(["${MATRIX[5]}_${PRECOND[5]}"]='-1')


NUM_GPUS=1
RES_COLLECTIVE="./cuda_a100_cg_results/"
echo "Setting CUDA devices"
devices=0
for ((i = 1; i < ${NUM_GPUS} ; i++)); do
	devices=${devices},${i}
done
export CUDA_VISIBLE_DEVICES=${devices}
sleep 1s
for idx in {1..5}; do
for precond in "${PRECOND[@]}"; do
	x=0
	for mtx in "${MATRIX[@]}"; do
		key="${mtx}_${precond}"
		if [[ ${ARGX[${key}]}  == '-1' ]]; then continue; fi
		echo "Preconditioner = ${precond}; Matrix = ${mtx}; Repetition: $idx"
		echo "Creating directories"
		sleep 1s
		mtxs=${mtx/\//_}
		dir_name="PC_${precond}_MTX_${mtxs}_REP_${idx}_CG"
		echo "Creating directory: " $dir_name
		mkdir -p ${dir_name}
		rm -f ${dir_name}/*
		cd ${dir_name}
		sleep 1s
		mkdir -p ${RES_HOME}
		mkdir -p ${RES_OUT_HOME}
		rm -f ${RES_HOME}/*
		rm -f ${RES_OUT_HOME}/*
		
		echo "Starting nvidia-smi"
		sleep 1s
		power_sids=()
		for ((i = 0; i < ${NUM_GPUS} ; i++)); do
			nvidia-smi -i ${i} --loop-ms=1000 --format=csv --query-gpu=power.draw,utilization.gpu,utilization.memory > ${RES_HOME}/gpu_${i}.txt &
			power_sids+=($!)
		done
		
		echo "Running the code"
		mtxr=${MTX_HOME}/${mtx}
		hrsxr=''
		if [[ ${RHS[$x]}  != '' ]]; then 
			hrsxr=${MTX_HOME}/${RHS[$x]}
		fi
		sleep 1s
		full_command="${SCRIPT} ${mtxr} ${precond} 1e-12 25000 ${ARGX[${key}]} ${hrsxr}"
		echo "CMD: " + $full_command
		${full_command} > ${RES_OUT_HOME}/output.txt
		
		echo "Killing all nvidia-smi"
		sleep 1s
		for sid in "${power_sids[@]}"; do
			kill ${sid}
		done
		pkill nvidia-smi
		
		echo "Done"
		sleep 5s
		cd ..
		let x++
	done
done
done

mkdir -p ${RES_COLLECTIVE}
rm -rf ${RES_COLLECTIVE}/*
mv PC_* ${RES_COLLECTIVE}
