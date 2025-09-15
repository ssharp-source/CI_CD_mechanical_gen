# CI_CD_mechanical_gen

### big idea
use github/gitlab to recursivly generate machincal hw through generative AI then QC with simulation and user feedback then feed the score back inot the start of the pipe


### tech stack
ci/cd kicked off with user prompt committed to branch.
at the hart a RAG set up with ollama under the hood with db with talored wrapper promts and Hardware db knoledge of known mechanizume and assembleis 
Openscad or similar as a STL solidmodle output frame work (think code to object) 
as for simulation inital though is physics engine in unity to simulate effects (gravity, collisions, ...) 



## use case 
scale comput based on task, llm using GPUs while openscad would be fine with cpu 
