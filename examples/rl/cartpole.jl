using POMDPs, Crux, Flux
import POMDPTools: Deterministic, ImplicitDistribution, EpsGreedyPolicy, LinearDecaySchedule
import QuickPOMDPs: QuickPOMDP
import ReinforcementLearningEnvironments: CartPoleEnv, state, act!, reset!, is_terminated, reward
import Random

# Cartpole - V0
# Convert ReinforcementLearningEnvironments.CartPoleEnv() to the POMDPs.jl interface
mdp = QuickPOMDP(
    actions = [1, 2],
    discount = 0.99f0,
    gen = function (s, a, rng)
        sp = deepcopy(s)
        act!(sp, a)
        o = state(sp)
        r = reward(sp)
        (;sp, o, r)
    end,
    initialstate = ImplicitDistribution((rng) -> CartPoleEnv()),
    isterminal = is_terminated,
    initialobs = s -> Deterministic(state(s))
)
POMDPs.gen(m::typeof(mdp), s, a) = POMDPs.gen(m, s, a, Random.default_rng())

as = actions(mdp)
S = state_space(mdp)

A() = DiscreteNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, length(as))), as)
V() = ContinuousNetwork(Chain(Dense(Crux.dim(S)..., 64, relu), Dense(64, 64, relu), Dense(64, 1)))

println("Solve with REINFORCE (~2 seconds)")
𝒮_reinforce = REINFORCE(π=A(), S=S, N=10000, ΔN=500, a_opt=(epochs=5,), interaction_storage=[])
@time π_reinforce = solve(𝒮_reinforce, mdp)

println("Solve with A2C (~8 seconds)")
𝒮_a2c = A2C(π=ActorCritic(A(), V()), S=S, N=10000, ΔN=500)
@time π_a2c = solve(𝒮_a2c, mdp)

println("Solve with PPO (~15 seconds)")
𝒮_ppo = PPO(π=ActorCritic(A(), V()), S=S, N=10000, ΔN=500)
@time π_ppo = solve(𝒮_ppo, mdp)

println("Solve with DQN (~12 seconds)")
𝒮_dqn = DQN(π=A(), S=S, N=10000, interaction_storage=[])
@time π_dqn = solve(𝒮_dqn, mdp)

println("Solve with SoftQLearning w/ varying α (~12 seconds)")
𝒮_sql = SoftQ(π=A(), α=Float32(0.1), S=S, N=10000, 
    ΔN=1, c_opt=(;epochs=5), interaction_storage=[])
@time π_sql = solve(𝒮_sql, mdp)

# Plot the learning curve
p = plot_learning([𝒮_reinforce, 𝒮_a2c, 𝒮_ppo, 𝒮_dqn, 𝒮_sql], title = "CartPole-V0 Training Curves", 
    labels = ["REINFORCE", "A2C", "PPO", "DQN", "SoftQ"])
Crux.savefig(p, "cartpole_training.pdf")

# Commented out broken code due to a lack of the `render` method for our CartPole environment
# Produce a gif with the final policy
# gif(mdp, π_ppo, "cartpole_policy.gif", max_steps=100)

## Optional - Save data for imitation learning
# using BSON
# s = Sampler(mdp, 𝒮_dqn.agent, max_steps=100, required_columns=[:t])
# 
# data = steps!(s, Nsteps=10000)
# sum(data[:r])/100
# data[:expert_val] = ones(Float32, 1, 10000)
# data[:a]
# 
# data = ExperienceBuffer(data)
# BSON.@save "examples/il/expert_data/cartpole.bson" data

println("Showing animation of PPO training outcome")

import Plots: plot

function action_choice(network, env)
    action_vals = network(convert(Vector{Float32}, state(env)))
    argmax(action_vals)
end

policy = π_ppo.A
max_sim_steps = 200
e = CartPoleEnv()
total_reward = 0f0
for i in 1:max_sim_steps+1
    a = action_choice(policy, e)
    act!(e, a)
    plot(e)
    global total_reward += reward(e)
    if is_terminated(e)
        break
    end
    sleep(0.05) # animation time step
end
@show total_reward

