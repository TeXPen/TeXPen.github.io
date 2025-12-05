Priority
- [ ] Gray out quantized selections for now (bad performance)
- [x] Unload model on page refresh, ensure never double loaded/memory leak etc.
- [ ] if the user draws new input or switches model, queue the inference, allow up to 3 seconds for previous one to finish, else skip to new

Bugs
- [ ] fix light mode

Behavior

Features
- [ ] add image upload tab
    - change loading UI to allow changing tabs while loading model, block canvas

Maintenance
- [ ] refactor and add tests