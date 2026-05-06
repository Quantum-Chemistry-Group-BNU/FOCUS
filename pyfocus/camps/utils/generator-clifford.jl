using NPZ
using QuantumClifford
using QuantumOptics
using Base

flag_run = true
flag_phase = false
n_qubits = 4
if flag_run

    folder_path = "./file"
    if !isdir(folder_path)
        mkdir(folder_path)
    end 

    if flag_phase
        data = collect(enumerate_phases(enumerate_cliffords(n_qubits)))
    else
        # data = collect(enumerate_cliffords(2))
		data = []
		for i in 1:1000
	        push!(data, enumerate_cliffords(n_qubits, i)) 
		end
    end
    clifford_data = cat([op.data for op in Operator.(data)]..., dims=3)  # shape: (4, 4, N)
    clifford_data = permutedims(clifford_data, (3, 1, 2))  # shape: (N, 4, 4)
    println(size(clifford_data))

    if flag_phase
        npzwrite("./file/clifford-4bits-operators-with-phase.npz", Dict("clifford_ops" => clifford_data))
    else
        npzwrite("./file/clifford-4bits-operators-random.npz", Dict("clifford_ops" => clifford_data))
    end

    function clean_pauli_string(s::String)
        subs = Dict('₀'=>'0', '₁'=>'1', '₂'=>'2', '₃'=>'3', '₄'=>'4',
                    '₅'=>'5', '₆'=>'6', '₇'=>'7', '₈'=>'8', '₉'=>'9')
        s1 = join([get(subs, c, c) for c in s])
        s2 = replace(s1, "⟼" => "->")
        s3 = replace(s2, "_" => "I")
        return s3
    end

    all_cliffords = data

    println("Save clifford")
    
    if flag_phase
        map_file = "./file/all_cliffords-with-phase.txt"
    else
        map_file = "./file/all_cliffords-random.txt"
    end
    open(map_file, "w") do io
        for (i, cl) in enumerate(all_cliffords)
            println(io, string(i-1))
            s = string(cl)
            s = clean_pauli_string(s)
            println(io, s)
            println(io)
        end
    end
end
