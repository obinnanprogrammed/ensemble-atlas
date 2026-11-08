## Overview
This is an app where users can search inside audio recordings using natural language or an example sound, backed by a recoverable pool of embedding workers. It returns the best matching recordings with the best matching chunks from each recording. The development specification is in README.md.

## Project Environment
- Python 3.12+
- Virtual Environment: eavenv

## General Guidelines
- Implement according to the specification in README.md.
- If anything is unclear, halt and request clarification.
- Request permission before installing any packages not specified in README.md.

## Project-Specific Guidelines (reiterated from README.md)
Treat the scope and ownership boundaries above as the starting contract. Keep changes small and runnable. Preserve the Python-first CLI workflow, implement recovery semantics explicitly, and update this README when an interface changes. Do not add Go, a web server, cloud infrastructure, or a vector database without a new project decision. Clearly separate implemented functionality from planned behavior in documentation and demos.
