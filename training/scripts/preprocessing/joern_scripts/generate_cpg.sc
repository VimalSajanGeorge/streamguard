// generate_cpg.sc — Joern Scala script for CPG generation
// Produces nodes.json and edges.json for a single C file.
// CRITICAL: run.ossdataflow is MANDATORY for DFG edges.
// Without it Joern produces AST+CFG only — no data-flow signal.

@main def main(inputFile: String, outputDir: String) = {
  importCode(inputFile)
  run.ossdataflow    // MANDATORY — enables DFG edges
  save

  // Export nodes and edges as JSON
  val nodesJson = cpg.graph.V.toJsonPretty
  val edgesJson = cpg.graph.E.toJsonPretty

  os.write(os.Path(outputDir) / "nodes.json", nodesJson)
  os.write(os.Path(outputDir) / "edges.json", edgesJson)
}
