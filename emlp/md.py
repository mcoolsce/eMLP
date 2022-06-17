import numpy as np
from yaff import *
import h5py as h5
from molmod.periodic import periodic
from .help_functions import XYZLogger


class ModelPart(ForcePart):
    def __init__(self, system, model, centers, efield = [0, 0, 0], log_name = 'md.xyz', nprint = 1, print_opt_steps = False):
        ForcePart.__init__(self, 'eMLP', system)
        self.system = system
        self.model = model
        self.step = 0
        self.nprint = nprint
        self.efield = efield
        self.log_name = log_name
        self.print_opt_steps = print_opt_steps
        if self.system.cell.nvec == 0:
            self.abs_centers = centers
        else:
            self.frac_centers = np.einsum('ij,jk', centers, np.linalg.inv(self.system.cell.rvecs / angstrom))
        self.previous_positions = None
        if not self.log_name is None:
            self.logger = XYZLogger(log_name)
            

    def _internal_compute(self, gpos, vtens): 
        numbers = self.system.numbers
        positions = self.system.pos / angstrom
        
        if self.system.cell.nvec == 0:
            rvec = np.eye(3) * 100
            centers = self.abs_centers
        else:
            rvec = self.system.cell.rvecs / angstrom  
            centers = np.einsum('ij,jk', self.frac_centers, rvec)
            
        if not self.previous_positions is None:
            max_disp = np.max(np.abs(positions - self.previous_positions)) * 3. # angstrom
        else:
            max_disp = 0.1
                 
        list_of_properties = ['energy']
        if not gpos is None:
            list_of_properties.append('forces')
        if not vtens is None:
            list_of_properties.append('vtens')
        output = self.model.compute(positions, numbers, centers, efield = self.efield, rvec = rvec, list_of_properties = list_of_properties, max_disp = max_disp, verbose=self.print_opt_steps)
        
        if not vtens is None:
            vtens[:, :] = output['vtens'] * electronvolt
        else:
            output['vtens'] = None
        if not gpos is None:
            gpos[:, :] = -output['forces'] / angstrom * electronvolt
        else:
            output['forces'] = None
            output['center_forces'] = None
        
        if self.system.cell.nvec == 0:
            self.abs_centers = output['centers'].copy()
        else:
            self.frac_centers = np.einsum('ij,jk', output['centers'], np.linalg.inv(rvec))
        self.previous_positions = positions.copy()
         
        self.step += 1
        if not self.log_name is None:
            if self.step % self.nprint == 0:
                self.logger.write(numbers, positions, energy = output['energy'], centers = output['centers'], rvec = rvec, vtens = output['vtens'],
                                  forces = output['forces'], center_forces = output['center_forces'], efield = self.efield)
        
        return output['energy'] * electronvolt
        
        
class AllElectronPart(ForcePart):
    def __init__(self, system, model, efield = [0, 0, 0], log_name = 'centers.xyz', nprint = 1):
        ForcePart.__init__(self, 'eMLP', system)
        self.system = system
        self.model = model
        self.efield = efield
        self.nprint = nprint
        self.step = 0
        self.log_name = log_name
        
        if not self.log_name is None:
            self.logger = XYZLogger(self.log_name)

    def _internal_compute(self, gpos, vtens):
        numbers = self.system.numbers[np.where(self.system.numbers != 99)]
        centers = self.system.pos[np.where(self.system.numbers == 99)] / angstrom
        positions = self.system.pos[np.where(self.system.numbers != 99)] / angstrom
        rvec = self.system.cell.rvecs / angstrom
        
        list_of_properties = ['energy']
        if not gpos is None:
            list_of_properties.append('forces')
        if not vtens is None:
            list_of_properties.append('vtens')
        output = self.model.compute_static(positions, numbers, centers, efield = self.efield, rvec = rvec, list_of_properties = list_of_properties)
        if not vtens is None:
            vtens[:, :] = output['vtens'] * electronvolt
        else:
            output['vtens'] = None
        if not gpos is None:
            gpos[:, :] = -np.concatenate((output['forces'], output['center_forces']), axis = 0) / angstrom * electronvolt
        else:
            output['forces'] = None
            output['center_forces'] = None
        
        self.step += 1
        if self.step % self.nprint == 0:
            if not self.log_name is None:
                self.logger.write(numbers, positions, energy = output['energy'], centers = centers, rvec = rvec, vtens = output['vtens'],
                                  forces = output['forces'], center_forces = output['center_forces'], efield = self.efield)

        return output['energy'] * electronvolt
        
    
def Optimize(model, positions, numbers, centers, rvec = np.eye(3) * 100, log = None, fullcell = False, method = QNOptimizer): 
    all_numbers = np.concatenate((numbers, 99 * np.ones(centers.shape[0], dtype=np.int)), axis = 0)
    all_positions = np.concatenate((positions, centers), axis = 0)
    system = System(all_numbers, all_positions * angstrom, rvecs = rvec.astype(np.float) * angstrom)
    
    ff = ForceField(system, [AllElectronPart(system, model, efield = [0, 0, 0], log_name = log, nprint = 1)])
    if fullcell:
        opt = method(FullCellDOF(ff, gpos_rms = 1e-07, grvecs_rms=1e-07))
    else:
        opt = method(CartesianDOF(ff, gpos_rms = 1e-07))
    try:
        opt.run()
    except RuntimeError as error:
        print(str(error))
    
    opt_centers = ff.system.pos[np.where(ff.system.numbers == 99)] / angstrom
    opt_positions = ff.system.pos[np.where(ff.system.numbers != 99)] / angstrom
    opt_rvec = ff.system.cell.rvecs / angstrom
    return opt_positions, opt_centers, opt_rvec
    
    
def NVE(system, model, steps, centers, efield = [0, 0, 0], nprint = 10, dt = 1, temp = 300, start = 0, name = 'md', screenprint = 1000, print_opt_steps = False):
    ff = ForceField(system, [ModelPart(system, model, centers, efield = efield, log_name = name + '.xyz', nprint = nprint, print_opt_steps = print_opt_steps)])
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
    
    
def NVT(system, model, steps, centers, efield = [0, 0, 0], nprint = 10, dt = 1, temp = 300, start = 0, name = 'md', screenprint = 1000, print_opt_steps = False):
    ff = ForceField(system, [ModelPart(system, model, centers, efield = efield, log_name = name + '.xyz', nprint = nprint, print_opt_steps = print_opt_steps)])
    thermo = NHCThermostat(temp = temp)
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, thermo, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
    

def NPT(system, model, steps, centers, efield = [0, 0, 0], nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', screenprint = 1000, pressure = 1e+05 * pascal, print_opt_steps = False):
    ff = ForceField(system, [ModelPart(system, model, centers, efield = efield, log_name = name + '.xyz', nprint = nprint, print_opt_steps = print_opt_steps)])
    thermo = NHCThermostat(temp = temp)
    baro = MTKBarostat(ff, temp = temp, press = pressure)
    tbc = TBCombination(thermo, baro)
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
    
    
def NVsigmaT(system, model, steps, centers, efield = [0, 0, 0], nprint = 10, dt = 1, temp = 300, start = 0, name = 'run', screenprint = 1000, pressure = 1e+05 * pascal, print_opt_steps = False):
    ff = ForceField(system, [ModelPart(system, model, centers, efield = efield, log_name = name + '.xyz', nprint = nprint, print_opt_steps = print_opt_steps)])
    thermo = NHCThermostat(temp = temp)
    baro = MTKBarostat(ff, temp = temp, press = pressure, vol_constraint = True)
    tbc = TBCombination(thermo, baro)
    f = h5.File(name + '.h5', mode = 'w')
    hdf5_writer = HDF5Writer(f, start = start, step = nprint)
    sl = VerletScreenLog(step = screenprint)
    verlet = VerletIntegrator(ff, dt * femtosecond, hooks = [sl, tbc, hdf5_writer], temp0 = temp)
    verlet.run(steps)
    f.close()
