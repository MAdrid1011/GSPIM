ThisBuild / scalaVersion := "2.13.14"
ThisBuild / organization := "org.gspim"
ThisBuild / version := "0.1.0"

libraryDependencies ++= Seq(
  "org.chipsalliance" %% "chisel" % "6.6.0",
  "edu.berkeley.cs" %% "chiseltest" % "6.0.0" % Test,
  "org.scalatest" %% "scalatest" % "3.2.19" % Test
)

addCompilerPlugin("org.chipsalliance" % "chisel-plugin_2.13.14" % "6.6.0")

Test / fork := true
