Experiments around the **Creator-Critic Flow** for LLM agents. Inspired by reinforcement learning paradigms, particularly **Actor-Critic** methods. Consists of a dynamic interplay between two roles.

## Posts

Write-ups stemming from these scripts:

- [`simple-creator-critic-sprite-sheet-requirements-generator.py`: designing a complex technical pipeline](https://www.kokutech.com/blog/gamedev/sprite-pipeline-with-llm-critics)
- [`simple-creator-critic-parenting-structures.py`: see if LLMs can produce parenting advice](https://www.kokutech.com/blog/growth/parental-structures-designed-by-llms)
- [`simple-creator-critic-environmental-solutions.py`: test if agents can arrive at environmental solutions](https://www.kokutech.com/blog/growth/can-llms-arrive-at-novel-environmental-solutions)
- [`simple-creator-critic*.py`: agent-critic with dual critic tests](https://www.kokutech.com/blog/growth/dual-critic-agent-architecture-results)
- [`creator-critic*.py`: first creator-critic tests](https://www.kokutech.com/blog/growth/creator-critic-agent-architecture-results)

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
