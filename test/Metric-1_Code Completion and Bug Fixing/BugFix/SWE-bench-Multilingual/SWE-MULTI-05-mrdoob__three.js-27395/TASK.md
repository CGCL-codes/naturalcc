# SWE-bench Multilingual task: mrdoob__three.js-27395

Resolve the GitHub issue below in the repository workspace. Follow existing conventions and add or update tests as appropriate. Do not use the reference patch; the official evaluator validates the patch in the repository's pinned environment.

## Issue

Class Sphere is missing property isSphere
### Description

Box3 has this.isBox3 = true;
Sphere does not have this.isSphere;

### Reproduction steps

I need it for 
if(collider.isSphere)

### Code

a = new Sphere()
if(a.isSphere){ debugger }

### Live example

no example

### Screenshots

_No response_

### Version

r159

### Device

_No response_

### Browser

_No response_

### OS

_No response_


## Repository and revision

- Repository: `mrdoob/three.js`
- Language: JavaScript
- Base commit: `b99b3fd955864f534b1e2649518112f7e50fef21`
- SWE-bench instance: `mrdoob__three.js-27395`

The selected record is metadata and task text. It does not contain a checked-out repository; use the official harness or fetch the repository at the exact base commit before asking the agent to edit files.
