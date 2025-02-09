Experiments around the **Creator-Critic Flow** for LLM agents. Inspired by reinforcement learning paradigms, particularly **Actor-Critic** methods. Consists of a dynamic interplay between two roles.

## Posts

Write-ups stemming from these scripts:

- [agent-critic with dual critic tests](https://www.kokutech.com/blog/growth/dual-critic-agent-architecture-results)
- [first creator-critic tests](https://www.kokutech.com/blog/growth/creator-critic-agent-architecture-results)

## Definitions

1. **Creator**  
   - This is the LLM acting as the **idea generator**.  
   - It produces responses, plans, or solutions based on a given prompt or task.  
   - The creator’s goal is to maximize novelty, creativity, or usefulness in its outputs.

2. **Critic**  
   - This is a secondary process (another LLM pass or a separate model) that **assesses the creator’s output**.  
   - It evaluates based on predefined criteria like coherence, correctness, ethical alignment, or efficiency.  
   - The critic provides feedback, assigns scores, or refines the output.

### **High-Level Flow of Interaction**
1. **Generation** – The Creator proposes an initial output.  
2. **Evaluation** – The Critic reviews it, scoring quality, consistency, or other metrics.  
3. **Iteration** – The Creator refines its output based on the Critic’s feedback.  
4. **Finalization** – Once a threshold is met, the best response is selected.
