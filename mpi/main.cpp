#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdlib>
#include <climits>
#include <vector>

#include <mpi.h>

//MPI_Send(
//    void* /* указатель на начало данных */,
//    int /* количество элементов */,
//    MPI_INT/MPI_DOUBLE/... /* тип элементов (из списка стандартных MPI) */,
//    int /* номер получателя */,
//    int /* тэг сообщения */,
//    MPI_COMM_WORLD /* коммуникатор */
//)

//MPI_Recv(
//    void* /* указатель на начало данных */,
//    int /* количество элементов */,
//    MPI_INT/MPI_DOUBLE/... /* тип элементов (из списка стандартных MPI) */,
//    0 /* номер отправителя */,
//    0 /* тэг сообщения */,
//    MPI_COMM_WORLD /* коммуникатор */,
//    MPI_Status* /* данные о параметрах сообщения (откуда, сколько, с каким тэгом) */
//)

struct SubsegmentInfo {
  double start;
  int steps;
};

double FuncVal(double x) {
  return 4 / (1 + x * x);
}

double CalcSum(double start, int steps, double step) {
  double res = 0;
  for (int i = 0; i < steps; ++i) {
    res += FuncVal(start) * step;
    start += step;
  }

  return res;
}

void SendSegment(int dest, double start, int steps) {
  SubsegmentInfo info = SubsegmentInfo {
      .start = start,
      .steps = steps
  };

  MPI_Send(
      &info,
      sizeof(SubsegmentInfo),
      MPI_CHAR,
      dest,
      0,
      MPI_COMM_WORLD
  );
}

SubsegmentInfo DivideSegment(int segments, double step, int procs) {
  int parts_per_one = segments / procs;
  int parts_per_one_inc = parts_per_one + 1;
  double seg_len = parts_per_one * step;
  double seg_len_inc = parts_per_one_inc * step;
  double cur_start = seg_len;

  int procs_without_inc = procs - segments % procs;

  for (int i = 1; i < procs_without_inc; ++i) {
    SendSegment(i, cur_start, parts_per_one);
    cur_start += seg_len;
  }

  for (int i = procs_without_inc; i < procs; ++i) {
    SendSegment(i, cur_start, parts_per_one_inc);
    cur_start += seg_len_inc;
  }

  SubsegmentInfo segment_for_0_proc = SubsegmentInfo {
      .start = 0,
      .steps = parts_per_one
  };

  return segment_for_0_proc;
}

void LoggedIntMain(int segments, double step, const std::string& out_dir) {
  int procs = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  SubsegmentInfo segment = DivideSegment(segments, step, procs);

  double multi_res = CalcSum(segment.start, segment.steps, step);
  double other_res = 0;

  MPI_Status status;
  std::vector<double> all_res(procs, multi_res);
  for (int i = 1; i < procs; ++i) {
    MPI_Recv(
        &other_res,
        1,
        MPI_DOUBLE,
        MPI_ANY_SOURCE,
        MPI_ANY_TAG,
        MPI_COMM_WORLD,
        &status
    );
    multi_res += other_res;
    all_res[status.MPI_SOURCE] = other_res;
  }

  // Print data.
  std::ofstream out(out_dir + "/integrals.txt", std::ios::out | std::ios::app);
  for (size_t i = 0; i < procs; ++i) {
    out << "value from proc #" << i << ": " << all_res[i] << '\n';
  }
  out << "---------------------------------------------\n";
  double mono_res = CalcSum(0, segments, step);
  out << "value of the integral (all procs): " << multi_res << '\n';
  out << "value of the integral (only 0 proc): " << mono_res << '\n';
  out << "values differs by " << std::abs(mono_res - multi_res) << '\n';
}

void LoggedTimeMain(int segments, double step, const std::string& out_dir) {
  double begin_time = 0;
  double end_time = 0;

  // Parallel execution.
  begin_time = MPI_Wtime();

  int procs = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  SubsegmentInfo segment = DivideSegment(segments, step, procs);

  double multi_res = CalcSum(segment.start, segment.steps, step);
  double other_res = 0;

  MPI_Status status;
  for (int i = 1; i < procs; ++i) {
    MPI_Recv(
        &other_res,
        1,
        MPI_DOUBLE,
        MPI_ANY_SOURCE,
        MPI_ANY_TAG,
        MPI_COMM_WORLD,
        &status
    );
    multi_res += other_res;
  }

  end_time = MPI_Wtime();
  double multi_secs = end_time - begin_time;

  // Mono execution.
  begin_time = MPI_Wtime();

  CalcSum(0, segments, step);

  end_time = MPI_Wtime();
  double mono_secs = end_time - begin_time;

  // Print acceleration.
  std::ofstream out(out_dir + "/acceleration.csv", std::ios::out | std::ios::app);
  out << procs << ";" << segments << ";" << mono_secs / multi_secs << '\n';
}

void MainProc(int segments, double step, const std::string& out_dir) {
  LoggedIntMain(segments, step, out_dir);
  LoggedTimeMain(segments, step, out_dir);
}

void CalculateInt(double step) {
  SubsegmentInfo info;
  MPI_Status status;

  MPI_Recv(
      &info,
      sizeof(SubsegmentInfo),
      MPI_CHAR,
      0,
      MPI_ANY_TAG,
      MPI_COMM_WORLD,
      &status
  );

  double res = CalcSum(info.start, info.steps, step);

  MPI_Send(
      &res,
      1,
      MPI_DOUBLE,
      0,
      0,
      MPI_COMM_WORLD
  );
}

void SanitizeCmdArgs(int argc, char** argv) {
  if (argc == 3) {
    return;
  }

  std::cout << "Number of processes and path to output dir must be passed as cmd argument.\n"
            << "Actual:\n"
            << "argc: " << argc << '\n'
            << "argv:\n";
  for (int i = 0; i < argc; ++i) {
    std::cout << i << ": " << argv[i] << '\n';
  }
  exit(1);
}

int main(int argc, char** argv) {
  SanitizeCmdArgs(argc, argv);

  MPI_Init(&argc, &argv);

  int segments = std::atoi(argv[1]);
  double step = 1.0 / segments;

  int proc_id = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

  if (proc_id == 0) {
    // Handle cmd args.
    SanitizeCmdArgs(argc, argv);
    char* real_path = new char[PATH_MAX];
    std::string out_dir = realpath(argv[2], real_path);
    delete[] real_path;

    // main logic
    MainProc(segments, step, out_dir);
  } else {
    CalculateInt(step); // for integral.txt
    CalculateInt(step); // for acceleration.csv
  }

  MPI_Finalize();
  return 0;
}