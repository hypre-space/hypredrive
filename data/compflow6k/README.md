This test case was generated using the GEOS application, specifically from commit hash
[d306d3f8](https://github.com/GEOS-DEV/GEOS/commit/d306d3f8ac8aac8a3cdf6778e3041bbd7ab4375d). Below,
you will find a copy of the input file that was used.

The linear system located in the `np1` directory was produced by executing GEOS with a
single process. Conversely, the system found in the `np4` directory resulted from running
GEOS with 4 processes, using the `-x 4` option to specify the process count.

```
<?xml version="1.0" ?>

<Problem>
  <Solvers
    gravityVector="{ 0.0, 0.0, 9.81 }">
    <CompositionalMultiphaseFVM
      name="compflow"
      logLevel="1"
      discretization="fluidTPFA"
      temperature="300"
      useMass="1"
      initialDt="1e4"
      solutionChangeScalingFactor="1"
      targetPhaseVolFractionChangeInTimeStep="0.1"
      maxCompFractionChange="0.2"
      targetRegions="{ region }">
      <NonlinearSolverParameters
        newtonTol="1e-3"
        newtonMaxIter="20"
        maxTimeStepCuts="10"
        lineSearchAction="Attempt"
        lineSearchMaxCuts="2"/>
      <LinearSolverParameters
        solverType="fgmres"
        preconditionerType="mgr"
        krylovAdaptiveTol="1"
        krylovWeakestTol="1.0e-3"
        krylovTol="1.0e-4"
        logLevel="50"/>
    </CompositionalMultiphaseFVM>
  </Solvers>

  <NumericalMethods>
    <FiniteVolume>
      <TwoPointFluxApproximation
        name="fluidTPFA"/>
    </FiniteVolume>
  </NumericalMethods>

  <ElementRegions>
    <CellElementRegion
      name="region"
      cellBlocks="{ block }"
      materialList="{ fluid, rock, relperm }"/>
  </ElementRegions>

  <Constitutive>
    <DeadOilFluid
      name="fluid"
      phaseNames="{ oil, water }"
      surfaceDensities="{ 800.0, 1022.0 }"
      componentMolarWeight="{ 114e-3, 18e-3 }"
      hydrocarbonFormationVolFactorTableNames="{ B_o_table }"
      hydrocarbonViscosityTableNames="{ visc_o_table }"
      waterReferencePressure="30600000.1"
      waterFormationVolumeFactor="1.03"
      waterCompressibility="0.00000000041"
      waterViscosity="0.0003"/>
    <CompressibleSolidConstantPermeability
      name="rock"
      solidModelName="nullSolid"
      porosityModelName="rockPorosity"
      permeabilityModelName="rockPermeability"/>
    <NullModel
      name="nullSolid"/>
    <PressurePorosity
      name="rockPorosity"
      defaultReferencePorosity="0.1463"
      referencePressure="1.0e7"
      compressibility="1.0e-10"/>
    <ConstantPermeability
      name="rockPermeability"
      permeabilityComponents="{ 6.7593e-14, 6.7593e-14, 6.7593e-15 }"/>
    <BrooksCoreyRelativePermeability
      name="relperm"
      phaseNames="{ oil, water }"
      phaseMinVolumeFraction="{ 0.0, 0.0 }"
      phaseRelPermExponent="{ 2.0, 2.0 }"
      phaseRelPermMaxValue="{ 1.0, 1.0 }"/>
  </Constitutive>

  <FieldSpecifications>
    <FieldSpecification
      name="permx"
      component="0"
      initialCondition="1"
      setNames="{ all }"
      objectPath="ElementRegions/region/block"
      fieldName="rockPermeability_permeability"
      functionName="permxFunc"
      scale="9.869233e-16"/>
    <FieldSpecification
      name="permy"
      component="1"
      initialCondition="1"
      setNames="{ all }"
      objectPath="ElementRegions/region/block"
      fieldName="rockPermeability_permeability"
      functionName="permyFunc"
      scale="9.869233e-16"/>
    <FieldSpecification
      name="permz"
      component="2"
      initialCondition="1"
      setNames="{ all }"
      objectPath="ElementRegions/region/block"
      fieldName="rockPermeability_permeability"
      functionName="permzFunc"
      scale="9.869233e-16"/>
    <FieldSpecification
      name="referencePorosity"
      initialCondition="1"
      setNames="{ all }"
      objectPath="ElementRegions/region/block"
      fieldName="rockPorosity_referencePorosity"
      functionName="poroFunc"
      scale="1.0"/>
    <FieldSpecification
      name="initialPressure"
      initialCondition="1"
      setNames="{ all }"
      objectPath="ElementRegions/region/block"
      fieldName="pressure"
      scale="4.1369e+7"/>
    <FieldSpecification
      name="initialComposition_oil"
      initialCondition="1"
      setNames="{ all }"
      objectPath="ElementRegions/region/block"
      fieldName="globalCompFraction"
      component="0"
      scale="0.9999"/>
    <FieldSpecification
      name="initialComposition_water"
      initialCondition="1"
      setNames="{ all }"
      objectPath="ElementRegions/region/block"
      fieldName="globalCompFraction"
      component="1"
      scale="0.0001"/>
    <SourceFlux
      name="sourceTerm"
      objectPath="ElementRegions/region/block"
      scale="-0.07279"
      component="1"
      setNames="{ source }"/>
    <FieldSpecification
      name="sinkPressure"
      setNames="{ sink1, sink2, sink3, sink4 }"
      objectPath="ElementRegions/region/block"
      fieldName="pressure"
      scale="2.7579e+7"/>
    <FieldSpecification
      name="sinkComposition_oil"
      setNames="{ sink1, sink2, sink3, sink4 }"
      objectPath="ElementRegions/region/block"
      fieldName="globalCompFraction"
      component="0"
      scale="0.9999"/>
    <FieldSpecification
      name="sinkComposition_water"
      setNames="{ sink1, sink2, sink3, sink4 }"
      objectPath="ElementRegions/region/block"
      fieldName="globalCompFraction"
      component="1"
      scale="0.0001"/>
  </FieldSpecifications>

  <Functions>
    <TableFunction
      name="permxFunc"
      inputVarNames="{elementCenter}"
      coordinateFiles="{xlin.geos,ylin.geos,zlin.geos}"
      voxelFile="permx.geos"
      interpolation="linear" />
    <TableFunction
      name="permyFunc"
      inputVarNames="{elementCenter}"
      coordinateFiles="{xlin.geos,ylin.geos,zlin.geos}"
      voxelFile="permy.geos"
      interpolation="linear" />
    <TableFunction
      name="permzFunc"
      inputVarNames="{elementCenter}"
      coordinateFiles="{xlin.geos,ylin.geos,zlin.geos}"
      voxelFile="permz.geos"
      interpolation="linear" />
    <TableFunction
      name="poroFunc"
      inputVarNames="{elementCenter}"
      coordinateFiles="{xlin.geos,ylin.geos,zlin.geos}"
      voxelFile="poro.geos"
      interpolation="linear" />
    <TableFunction
      name="B_o_table"
      coordinateFiles="{ pres_pvdo.txt }"
      voxelFile="B_o_pvdo.txt"
      interpolation="linear"/>
    <TableFunction
      name="visc_o_table"
      coordinateFiles="{ pres_pvdo.txt }"
      voxelFile="visc_pvdo.txt"
      interpolation="linear"/>
  </Functions>

  <Events
    maxTime="1e7">
    <PeriodicEvent
      name="solverApplications"
      maxEventDt="1e6"
      target="/Solvers/compflow"/>
  </Events>

  <Mesh>
    <InternalMesh
      name="mesh"
      elementTypes="{ C3D8 }"
      xCoords="{ 0, 365.76 }"
      yCoords="{ 0, 670.56 }"
      zCoords="{ 0, 51.816 }"
      nx="{ 15 }"
      ny="{ 25 }"
      nz="{ 5 }"
      cellBlockNames="{ block }"/>
  </Mesh>

  <Geometry>
    <Box
      name="source"
      xMin="{ 182.85, 335.25, -0.01 }"
      xMax="{ 189.00, 338.35, 51.90 }"/>
    <Box
      name="sink1"
      xMin="{ -0.01, -0.01, -0.01 }"
      xMax="{ 6.126, 3.078, 51.90 }"/>
    <Box
      name="sink2"
      xMin="{ -0.01, 667.482, -0.01 }"
      xMax="{ 6.126, 670.60, 51.90 }"/>
    <Box
      name="sink3"
      xMin="{ 359.634, -0.01, -0.01 }"
      xMax="{ 365.8, 3.048, 51.90 }"/>
    <Box
      name="sink4"
      xMin="{ 359.634, 667.482, -0.01 }"
      xMax="{ 365.8, 670.60, 51.90 }"/>
  </Geometry>
</Problem>
```
