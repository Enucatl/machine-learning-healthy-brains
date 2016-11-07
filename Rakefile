require "time"
now = Time.now.to_i
user = ENV["USER"]
id = "#{user}-#{now}"
project = "ml-healthy-brains"
bucket = "gs://mlp2-data"

jobname = "#{project}-#{id}"

namespace :frontal_thickness do

  output = "#{bucket}/frontal_thickness/frontal_thickness-#{id}"
  desc "calculate the frontal thickness on the cloud with only two test files"
  task :cloud_smaller do
    sh [
      "python frontal_thickness.py",
      "--project #{project}",
      "--job_name #{jobname}",
      "--runner BlockingDataflowPipelineRunner",
      "--max_num_workers 24",
      "--autoscaling_algorithm THROUGHPUT_BASED",
      "--staging_location #{bucket}/staging",
      "--temp_location #{bucket}/temp",
      "--output #{output}",
      "--zone europe-west1-c",
      "--disk_size_gb 100",
      "--setup_file ./setup.py",
      "--input \"#{bucket}/set_train/train_10[01].nii\"",
    ].join(" ")
  end

  desc "calculate the frontal thickness on the cloud with all oversampled files"
  task :cloud_zoom do
    sh [
      "python frontal_thickness.py",
      "--project #{project}",
      "--job_name #{jobname}",
      "--runner DataflowPipelineRunner",
      "--max_num_workers 24",
      "--autoscaling_algorithm THROUGHPUT_BASED",
      "--staging_location #{bucket}/staging",
      "--temp_location #{bucket}/temp",
      "--output #{output}",
      "--zone europe-west1-c",
      "--disk_size_gb 100",
      "--setup_file ./setup.py",
      "--input \"#{bucket}/set_train/*.nii\"",
      "--zoom 3",
    ].join(" ")
  end

  desc "calculate the frontal thickness on the cloud with all files"
  task :cloud do
    sh [
      "python frontal_thickness.py",
      "--project #{project}",
      "--job_name #{jobname}",
      "--runner DataflowPipelineRunner",
      "--max_num_workers 24",
      "--autoscaling_algorithm THROUGHPUT_BASED",
      "--staging_location #{bucket}/staging",
      "--temp_location #{bucket}/temp",
      "--output #{output}",
      "--zone europe-west1-c",
      "--disk_size_gb 100",
      "--setup_file ./setup.py",
      "--input \"#{bucket}/set_*/*.nii\"",
    ].join(" ")
  end

  desc "ls output from the cloud"
  task :ls do
    sh "gsutil ls #{bucket}/frontal_thickness/"
  end

  desc "calculate the frontal thickness locally with all files"
  task :local_all do
    sh [
      "python frontal_thickness.py",
      "--input \"data/set_*/*.nii\"",
    ].join(" ")
  end

  desc "calculate the frontal thickness locally with two files only"
  task :local_small do
    sh [
      "python frontal_thickness.py",
    ].join(" ")
  end

end


namespace :cloud do
end
