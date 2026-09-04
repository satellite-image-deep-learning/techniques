# Repository guidance

## Adding a new README entry

- Check that the project is not already listed.
- Add it to the most specific relevant section in `README.md`, normally at the end of that section.
- Use the existing list format, with a blank line between entries:

  ```markdown
  - [Project name](https://canonical-project-url) -> Short factual description
  ```

- Always include a concise description explaining what the project, code, or dataset does.
- If the repository implements or accompanies a paper, include the paper title. Prefer this wording:

  ```markdown
  - [Project name](https://canonical-project-url) -> code for paper: Paper Title
  ```

- Preserve the project's official name and capitalization, and link directly to its canonical repository or project page.
- Do not reformat or alter unrelated entries.
- Before finishing, run `git diff --check` and review the README diff for correct placement and formatting.
