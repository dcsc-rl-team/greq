import jax
import jax.numpy as jnp
import jax.random as jrd
from flax import struct
from typing import Any
import chex
from chex import dataclass
import rlax 
import tyro

from jaxtor.env.tabular import TabularEnv
from jaxdp.typehints import F, QType, StaticMeta

class q_learning_async:

    @dataclass
    class alg_State:
        q_vals: QType
        gamma: jnp.ndarray
        c: QType

    def init(mdp: TabularEnv.State, gamma: jnp.ndarray) -> "q_learning_async.alg_State":
        
        # q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
        #                      dtype = 'float', minval =0.0, maxval = 1.0) # (a, s)
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        c = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s)

        return q_learning_async.alg_State(q_vals=q_vals, gamma=gamma, c=c)
    
    def update(state: "q_learning_async.alg_State", trans: Any, alpha, alpha_power, beta_power, k:int)  -> "q_learning_async.alg_State": 
        
        c_new = state.c.at[trans.act, trans.obs].add(1)
        discount = jnp.where(trans.term, 0.0, state.gamma)
        td_err = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )
        # alpha = 1/(c_new[trans.act, trans.obs]**alpha_power)
        next_q = state.q_vals.at[trans.act, trans.obs].add(alpha * td_err)

        return state.replace(q_vals=next_q, c=c_new)

class speedy_q_learning_async:

    @dataclass
    class alg_State:
        q_vals: QType
        prev_q: QType
        gamma: jnp.ndarray
        c: QType

    def init(mdp: TabularEnv.State, gamma: jnp.ndarray) -> "speedy_q_learning_async.alg_State":
        
        # q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
        #                      dtype = 'float', minval =0.0, maxval = 1.0) # (a, s)
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        c = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s)

        return speedy_q_learning_async.alg_State(q_vals=q_vals, prev_q=q_vals, gamma=gamma, c=c)
    
    def update(state: "speedy_q_learning_async.alg_State", trans: Any, alpha, alpha_power, beta_power, k:int)  -> "speedy_q_learning_async.alg_State": 
        
        c_new = state.c.at[trans.act, trans.obs].add(1)
        discount = jnp.where(trans.term, 0.0, state.gamma)
        prev_q = state.prev_q
        q_vals = state.q_vals
        prev_BE = trans.rew + discount*(jnp.max(prev_q[:, trans.nobs], axis=0))
        BE = trans.rew + discount*(jnp.max(q_vals[:, trans.nobs], axis=0))
        sql_td_error = alpha * (prev_BE - q_vals[trans.act, trans.obs]) + (1 - alpha) * (BE - prev_BE)
        # alpha = 1/(c_new[trans.act, trans.obs]**alpha_power)
        next_q = q_vals.at[trans.act, trans.obs].add(sql_td_error)

        return state.replace(q_vals=next_q, prev_q = q_vals, c=c_new)

class nesa_q_learning_async:

    @dataclass
    class alg_State:
        q_vals: jnp.ndarray
        prev_q: jnp.ndarray
        gamma: jnp.ndarray
        c: jnp.ndarray

    @staticmethod
    def init(mdp: Any, gamma: jnp.ndarray) -> "nesa_q_learning_async.alg_State":
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        c = jnp.zeros((mdp.action_size, mdp.state_size))
        return nesa_q_learning_async.alg_State(
            q_vals=q_vals, prev_q=q_vals, gamma=gamma, c=c
        )

    @staticmethod
    def update(state: "nesa_q_learning_async.alg_State", trans: Any, alpha: float, alpha_power: float, beta_power: float, k: int) -> "nesa_q_learning_async.alg_State":
        s, a = trans.obs, trans.act
        s_next = trans.nobs
        zeta = 1.0
        discount = jnp.where(trans.term, 0.0, state.gamma)

        # 1. Update visit counts
        c_new = state.c.at[a, s].add(1)

        # 2. Greedy next action w.r.t current Q
        nact = jnp.argmax(state.q_vals[:, s_next], axis=0)

        # 3. Previous parameter difference Delta Q_n = Q_n - Q_{n-1}
        delta_prev = state.q_vals - state.prev_q

        # 4. Instantaneous momentum projection along (I + zeta * A_{n+1})
        # For entry (a, s): Delta Q_n(a, s) + zeta * (gamma * Delta Q_n(nact, s_next) - Delta Q_n(a, s))
        proj_momentum = (
            delta_prev[a, s]
            + zeta * (discount * delta_prev[nact, s_next] - delta_prev[a, s])
        )

        # 5. TD error 
        td_error = (
            trans.rew
            + discount * state.q_vals[nact, s_next]
            - state.q_vals[a, s]
        )

        # 6. NeSA step update for the visited coordinate
        step_diff = proj_momentum + alpha * zeta * td_error

        # 7. Apply sparse update at (a, s) only; unvisited states remain unchanged
        next_q = state.q_vals.at[a, s].add(step_diff)

        return state.replace(q_vals=next_q, prev_q=state.q_vals, c=c_new)
    
class zap_q_learning_async:

    @dataclass
    class alg_State:
        q_vals: QType
        gamma: jnp.ndarray
        matrix_gain: Any
        c: QType

    def init(mdp: TabularEnv.State, gamma: jnp.ndarray) -> "zap_q_learning_async.alg_State":
        
        # q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
        #                      dtype = 'float', minval =0.0, maxval = 1.0) # (a, s)
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        matrix_gain = jnp.eye(mdp.action_size*mdp.state_size)
        c = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s)

        return zap_q_learning_async.alg_State(q_vals=q_vals, gamma=gamma, matrix_gain=matrix_gain, c=c)
    
    def update(state: "zap_q_learning_async.alg_State", trans: Any, alpha, alpha_power, beta_power, k:int)  -> "zap_q_learning_async.alg_State": 
        
        c_new = state.c.at[trans.act, trans.obs].add(1)
        beta = 1/((1+c_new[trans.act, trans.obs])**beta_power)
        # beta = (1/(2 + k)**beta_power)
        discount = jnp.where(trans.term, 0.0, state.gamma)
        nact = jnp.argmax(state.q_vals[:, trans.nobs], axis=0)

        # \hat{A_{k+1}} approximation. Here \hat{A_{k+1}} is (I - \gamma P_hat) 
        A, S = state.q_vals.shape
        E_zeros = jnp.zeros([A, S, A, S]) # 4-D Tensor
        E = E_zeros.at[trans.act, trans.obs, trans.act, trans.obs].set(1) # setting one at (a,s,a,s)
        E_reshape = E.reshape(A*S, A*S) # reshaping it to (as, as)
        P_hat = jnp.zeros((A, S, A, S))
        P_hat_upd = P_hat.at[trans.act, trans.obs, nact, trans.nobs].set(1.0) # setting one at (a,s,a',s^+)
        step_matrix_gain = (E_reshape - state.gamma*(P_hat_upd.reshape(A*S, A*S))) # (E - \gamma P)
        next_matrix_gain = state.matrix_gain + beta * (step_matrix_gain - state.matrix_gain) # SA update of (I - \gamma P) and shape is (as, as)
        
        td_error = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )

        g= jnp.zeros((A, S))
        g_flat = (g.at[trans.act, trans.obs].set(td_error)).flatten()
        pre_td_err = jnp.linalg.solve(next_matrix_gain, g_flat).reshape(A, S)
        alpha = 1/(1+c_new[trans.act, trans.obs])**alpha_power

        new_q = state.q_vals + alpha * pre_td_err

        return state.replace(q_vals=new_q, matrix_gain=next_matrix_gain, c=c_new)

class zap_q_learning_od_async:

    @dataclass
    class alg_State:
        q_vals: jnp.ndarray            # Shape: (A, S)
        gamma: jnp.ndarray             # Discount factor
        matrix_gain: jnp.ndarray       # Shape: (A*S, A*S), tracks empirical Jacobian A
        c: jnp.ndarray                 # Shape: (A, S) visit counter
        # Accumulators for the current batch of length N:
        batch_f: jnp.ndarray           # Accumulated TD error vector, shape: (A*S,)
        batch_grad_f: jnp.ndarray      # Accumulated Jacobian matrix, shape: (A*S, A*S)
        beta_prod: jnp.ndarray         # Scalar accumulator for \prod (1 - \beta_j)
        batch_step: int                # Step counter within current batch [0, N-1]
        batch_size: int                # N >= (A*S)^2 for amortized O(d) per step
        batch_idx: int                 # Outer batch counter i

    def init(mdp: Any, gamma: jnp.ndarray) -> "zap_q_learning_od_async.alg_State":
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        d = mdp.action_size * mdp.state_size
        # To make amortized computation O(d) per step with an O(d^3) solve, N must be >= d^2
        N = 770 
        # Initializing matrix_gain as -I ensures negative definiteness (Hurwitz)
        matrix_gain = -jnp.eye(d)
        c = jnp.zeros((mdp.action_size, mdp.state_size))

        return zap_q_learning_od_async.alg_State(
            q_vals=q_vals,
            gamma=gamma,
            matrix_gain=matrix_gain,
            c=c,
            batch_f=jnp.zeros((d,)),
            batch_grad_f=jnp.zeros((d, d)),
            beta_prod=jnp.array(1.0),
            batch_step=0,
            batch_size=N,
            batch_idx=0
        )

    def update(state: "zap_q_learning_od_async.alg_State", trans: Any, alpha: float, alpha_power: float, beta_power: float, k: int) -> "zap_q_learning_od_async.alg_State":
        
        A, S = state.q_vals.shape
        d = A * S
        c_new = state.c.at[trans.act, trans.obs].add(1)
        discount = jnp.where(trans.term, 0.0, state.gamma)
        
        # Step-size for the fast scale at step k: beta_k = 1 / (k + 2)^beta_power
        beta = 1.0 / (2.0 + k)**beta_power

        # Inner transition evaluation using current batch parameter theta_{iN}
        nact = jnp.argmax(state.q_vals[:, trans.nobs], axis=0)
        td_error = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )

        idx_curr = trans.act * S + trans.obs
        idx_next = nact * S + trans.nobs

        # Accumulate \hat{f}(\theta_{iN}) and empirical Jacobian \nabla \hat{f}(\theta_{iN})
        # Note: rlax.q_learning computes (r + gamma * max Q - Q), consistent with \bar{f}
        batch_f_new = state.batch_f.at[idx_curr].add(td_error)
        batch_grad_f_new = state.batch_grad_f.at[idx_curr, idx_curr].add(-1.0)
        batch_grad_f_new = batch_grad_f_new.at[idx_curr, idx_next].add(discount)
        prod_new = state.beta_prod * (1.0 - beta)

        is_batch_end = (state.batch_step + 1) == state.batch_size

        def _batch_update(_):
            # Normalize batch averages
            f_hat = batch_f_new / state.batch_size
            grad_f_hat = batch_grad_f_new / state.batch_size
            hat_beta = 1.0 - prod_new

            # Update matrix gain \hat{A}_{(i+1)N}
            next_gain = state.matrix_gain + hat_beta * (grad_f_hat - state.matrix_gain)

            # Slow time-scale step size for batch i: alpha_{i+1} = alpha / (i + 2)^alpha_power
            alpha_i = 1 / (2.0 + state.batch_idx)**alpha_power

            # Ridge regularization (projection) prevents numerical singularities:
            # G^* = - (next_gain - eps * I)^{-1}
            # eps_reg = 1e-4
            # reg_gain = next_gain - eps_reg * jnp.eye(d)

            # Parameter update:
            # theta_{(i+1)N} = theta_{iN} + alpha * G^* f_hat = theta_{iN} - alpha * (A)^{-1} f_hat
            # Solving (next_gain) * dir = f_hat gives dir = A^{-1} f_hat
            newton_dir = jnp.linalg.solve(next_gain, f_hat)
            new_q = state.q_vals - alpha_i * newton_dir.reshape(A, S)

            return (
                new_q,
                next_gain,
                jnp.zeros((d,)),
                jnp.zeros((d, d)),
                jnp.array(1.0),
                0,
                state.batch_idx + 1
            )

        def _within_batch(_):
            return (
                state.q_vals,
                state.matrix_gain,
                batch_f_new,
                batch_grad_f_new,
                prod_new,
                state.batch_step + 1,
                state.batch_idx
            )

        (new_q, next_matrix_gain, b_f, b_grad, b_prod, next_b_step, next_b_idx) = jax.lax.cond(
            is_batch_end, _batch_update, _within_batch, operand=None
        )

        return state.replace(
            q_vals=new_q,
            matrix_gain=next_matrix_gain,
            c=c_new,
            batch_f=b_f,
            batch_grad_f=b_grad,
            beta_prod=b_prod,
            batch_step=next_b_step,
            batch_idx=next_b_idx
        )

class zap_zero_q_learning_async:

    @dataclass
    class alg_State:
        q_vals: jnp.ndarray
        gamma: jnp.ndarray
        w: jnp.ndarray
        z: jnp.ndarray
        c: jnp.ndarray

    def init(mdp: Any, gamma: jnp.ndarray) -> "zap_zero_q_learning_async.alg_State":
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        w = jnp.zeros((mdp.action_size, mdp.state_size))
        z = jnp.zeros((mdp.action_size, mdp.state_size))
        c = jnp.zeros((mdp.action_size, mdp.state_size), dtype=jnp.int32)

        return zap_zero_q_learning_async.alg_State(
            q_vals=q_vals,
            gamma=gamma,
            w=w,
            z=z,
            c=c,
        )

    def update(state: "zap_zero_q_learning_async.alg_State", trans: Any, alpha, alpha_power, beta_power, k: int) -> "zap_zero_q_learning_async.alg_State":

        m_scalar = 1.0
        curr_idx = (trans.act, trans.obs)
        c_new = state.c.at[curr_idx].add(1)
 
        # Fast time-scale step size
        # beta = 1.0 / ((2.0 + k) ** beta_power)
        beta = 1/(c_new[trans.act, trans.obs]**beta_power)
        alpha = 1/(c_new[trans.act, trans.obs]**alpha_power)
 
        discount = jnp.where(trans.term, 0.0, state.gamma)
        nact = jnp.argmax(state.q_vals[:, trans.nobs], axis=0)
        next_idx = (nact, trans.nobs)
 
        # Temporal difference: D_{k+1}
        td_error = rlax.q_learning(
                    state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
                )
       
        # Fast tracking variables y_k = q_k + z_k
        y_curr = state.q_vals[curr_idx] + state.z[curr_idx]
        y_next = state.q_vals[next_idx] + state.z[next_idx]
        delta_E_dot_y = discount * y_next - y_curr
 
        w_curr = state.w[curr_idx]
 
        # Q function update on visited pair
        new_q = state.q_vals.at[curr_idx].add(-alpha * y_curr)
 
        # Fast error update: w_{k+1} = w_k - beta * [A_{k+1} y_k - f_{k+1}]
        # Active only on curr_idx: drift is (delta_E_dot_y - td_error)
        new_w = state.w.at[curr_idx].add(-beta * (delta_E_dot_y - td_error))
 
        # Fast tracking update: z_{k+1} = z_k - beta * [z_k - L_w]
        # Apply local decay -beta * z(curr_idx) and local feedback -beta * m * w_curr
        new_z = state.z.at[curr_idx].add(-beta * (state.z[curr_idx] + m_scalar * w_curr))
 
        # Inject forward bootstrap feedback +beta * m * gamma * w_curr at next_idx
        new_z = new_z.at[next_idx].add(beta * m_scalar * discount * w_curr)
 
        return state.replace(
            q_vals=new_q,
            w=new_w,
            z=new_z,
            c=c_new,
        )


class pre_cond_q_learning_async:

    @dataclass
    class alg_State:
        q_vals: QType
        gamma: jnp.ndarray
        matrix_gain: Any
        P_hat: Any
        c: QType

    def init(mdp: TabularEnv.State, gamma: jnp.ndarray) -> "pre_cond_q_learning_async.alg_State":
        
        # q_vals = jrd.uniform(key, (mdp.action_size,mdp.state_size), 
        #                      dtype = 'float', minval =0.0, maxval = 1.0) # (a, s) 
        q_vals = jnp.zeros((mdp.action_size, mdp.state_size))
        matrix_gain = jnp.eye(mdp.action_size*mdp.state_size) # \hat{A}_{0}^{-1} (as, as)
        P_hat = jnp.zeros((mdp.action_size*mdp.state_size, mdp.action_size*mdp.state_size)) # (as, as)
        c = jnp.zeros((mdp.action_size, mdp.state_size)) # (a, s)

        return pre_cond_q_learning_async.alg_State(q_vals=q_vals, gamma=gamma, matrix_gain=matrix_gain, P_hat=P_hat,
                                                   c=c)
    
    def update(state: "pre_cond_q_learning_async.alg_State", trans: Any, alpha, alpha_power, beta_power, k:int)  -> "pre_cond_q_learning_async.alg_State": 

        discount = jnp.where(trans.term, 0.0, state.gamma)
        nact = jnp.argmax(state.q_vals[:, trans.nobs]) 
        c_new = (state.c.at[trans.act, trans.obs].add(1))
        
        # Approximating \hat{P} 
        A, S = state.q_vals.shape
        E_zeros = jnp.zeros([A, S]) # (a,s)
        E_k = E_zeros.at[trans.act, trans.obs].set(1) # setting 1 at (a,s)
        E_k_next = E_zeros.at[nact, trans.nobs].set(1) # setting 1 at (a',s^+)
        e_k = E_k.flatten() # reshaping it to (as, )
        e_k_next = E_k_next.flatten() # reshaping it to (as, )
        P_reshape = state.P_hat.reshape(A, S, A, S) # 4-D Tensor
        pdf_tens = P_reshape[trans.act, trans.obs, :, :] # 2-D Tensor
        pdf = pdf_tens.reshape(A*S) # \hat{P_{k}}((s,a), :)
        diff = e_k_next - pdf # (e_{k+1} - p_k(s,a))
        P_diff = jnp.einsum("i,j->ij", e_k, diff) # e_{k} * (e_{k+1}^T - p_k(s,a)^T)
        beta = 1/(c_new[trans.act, trans.obs]**beta_power)
        upd = beta * P_diff
        P_hat_new = state.P_hat + upd
        
        # Approximating \hat{A}_{k+1}^{-1} = (I - \gamma \hat{P}_{k+1})^{-1}
        u = -state.gamma * beta * e_k  # Shape (AS,)
        inv_u = jnp.dot(state.matrix_gain, u)      # (AS,)
        v_inv = jnp.dot(diff, state.matrix_gain)      # (AS,)
        den = 1.0 + jnp.dot(diff, inv_u) 
        num = jnp.outer(inv_u, v_inv)
        matrix_gain_new = state.matrix_gain - (num / den) # Woodsbury inversion formula

        td_error = rlax.q_learning(
            state.q_vals[:, trans.obs], trans.act, trans.rew, discount, state.q_vals[:, trans.nobs]
        )
        td= jnp.zeros([A, S])
        td_flat = (td.at[trans.act, trans.obs].set(td_error)).flatten()
        gain = (jnp.einsum("ij,j->i", matrix_gain_new, td_flat)).reshape(A, S)
        alpha = 1/(c_new[trans.act, trans.obs]**alpha_power)
        new_q = state.q_vals + alpha * gain

        return state.replace(q_vals=new_q, matrix_gain=matrix_gain_new, P_hat=P_hat_new, c=c_new)