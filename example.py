import numpy as np
import code.agents.asynchronous_dp as asyndp
import code.envs.nsbridge_v0 as nsb
import rats


def add_action(actions, action):
    if action == 0:
        actions.append('Left')
    elif action == 1:
        actions.append('Down')
    elif action == 2:
        actions.append('Right')
    elif action == 3:
        actions.append('Up')
    else:
        print('Error')
        exit()
    return actions


def main():
    np.random.seed(1993)

    # Parameters
    env = nsb.NSBridgeV0()
    env.set_epsilon(1.0)
    depth = 4
    # agent = rats.RATS(env.action_space, gamma=0.9, max_depth=depth)
    agent = asyndp.AsynDP(env.action_space, gamma=0.9, max_depth=depth, is_model_dynamic=False)

    agent.display()

    # Run
    for i in range(100):
        render = False
        actions = []
        done = False
        if render:
            env.render()
        timeout = 10
        undiscounted_return, total_time, discounted_return = 0.0, 0, 0.0
        env.reset()
        for t in range(timeout):
            action = agent.act(env, done)
            actions = add_action(actions, action)
            _, reward, done, _ = env.step(action)
            undiscounted_return += reward
            discounted_return += (agent.gamma ** t) * reward
            if render:
                print()
                env.render()
            if (t + 1 == timeout) or done:
                total_time = t + 1
                break
        print('End of episode')
        print('Total time          :', total_time)
        print('Actions             :', actions)
        print('Discounted return   :', discounted_return)
        print('Un-discounted return :', undiscounted_return)


if __name__ == "__main__":
    main()
