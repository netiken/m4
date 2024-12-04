// main.cpp

#include <iostream>
#include <vector>

// Function declaration (only declare, do not define)
extern "C" int interactive_inference(
    float* size,
    float* fat,
    float* i_fct,
    float* fct,
    float* sldn_flowsim_total,
    int n_flows_total,
    float* res_fct,
    float* res_sldn,
    int nhosts = 21,
    int lr = 10,
    int n_flows_active_max = 2000,
    int gpu_id = 1
);

// Main function to test the interactive_inference function
int main(int argc, char* argv[]) {
    int n_flows_total = 100; // Example number of flows
    if (argc > 1) {
        n_flows_total = std::stoi(argv[1]);
    }
    std::cout << "Number of flows: " << n_flows_total << std::endl;

    // Allocate input arrays
    std::vector<float> size(n_flows_total, 1000.0f);  // Flow sizes
    std::vector<float> fat(n_flows_total);            // Flow arrival times
    std::vector<float> i_fct(n_flows_total, 100.0f);  // Ideal flow completion times
    std::vector<float> fct(n_flows_total, 200.0f);    // Actual flow completion times
    std::vector<float> sldn_flowsim_total(n_flows_total, 1.0f); // Flow simulation slowdown

    // Initialize flow arrival times (e.g., flows arrive every 10 units)
    for (int i = 0; i < n_flows_total; ++i) {
        fat[i] = i * 10.0f;
    }

    // Allocate output arrays
    std::vector<float> res_fct(n_flows_total * 2, 0.0f);   // Each flow has 2 values (predicted and actual FCT)
    std::vector<float> res_sldn(n_flows_total * 2, 0.0f);  // Each flow has 2 values (predicted and actual SLDN)

    // Call the interactive_inference function
    int result = interactive_inference(
        size.data(),
        fat.data(),
        i_fct.data(),
        fct.data(),
        sldn_flowsim_total.data(),
        n_flows_total,
        res_fct.data(),
        res_sldn.data(),
        21,     // nhosts
        10,     // lr
        2000,    // n_flows_active_max
        1       // GPU ID
    );

    if (result == 0) {
        // Process and display the results
        std::cout << "First 10 FCT Results:\n";
        for (int i = 0; i < 10 && i < n_flows_total; ++i) {
            std::cout << "Flow ID " << i
                      << ": Predicted FCT = " << res_fct[i * 2]
                      << ", Actual FCT = " << res_fct[i * 2 + 1]
                      << std::endl;
        }

        std::cout << "\nFirst 10 SLDN Results:\n";
        for (int i = 0; i < 10 && i < n_flows_total; ++i) {
            std::cout << "Flow ID " << i
                      << ": Predicted SLDN = " << res_sldn[i * 2]
                      << ", Actual SLDN = " << res_sldn[i * 2 + 1]
                      << std::endl;
        }
    } else {
        std::cerr << "An error occurred during inference." << std::endl;
    }

    return 0;
}
