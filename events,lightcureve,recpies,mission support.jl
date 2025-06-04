using RecipesBase
using Statistics
using LinearAlgebra
using FITSIO, RecipesBase, StatsBase
using DocStringExtensions

"""
    FITSMetadata{H}

Metadata associated with a FITS or events file.

$(FIELDS)
"""
struct FITSMetadata{H}
    "Path to the FITS file"
    filepath::String
    "HDU index that the metadata was read from"
    hdu::Int
    "Units of energy (currently just ENERGY or PI or PHA)"
    energy_units::Union{Nothing,String}
    "Extra columns that were requested during read"
    extra_columns::Dict{String,Vector}
    "FITS headers from the selected HDU"
    headers::H
end

function Base.show(io::IO, ::MIME"text/plain", m::FITSMetadata)
    println(io, "FITSMetadata for $(basename(m.filepath))[$(m.hdu)] with $(length(m.extra_columns)) extra column(s)")
end


"""
    EventList{TimeType, MetaType <: FITSMetadata}

Container for an events list. Generally should not be directly constructed, but
read from file using [`readevents`](@ref).

$(FIELDS)
"""
struct EventList{TimeType<:AbstractVector, MetaType<:FITSMetadata}
    "Vector with recorded times"
    times::TimeType
    "Vector with recorded energies (else `nothing`)"
    energies::Union{Nothing,TimeType}
    "Metadata from FITS file"
    meta::MetaType
end

# Simple constructor for testing without FITS files
function EventList(times::Vector{T}, energies::Union{Nothing,Vector{T}}=nothing) where T
    dummy_meta = FITSMetadata(
        "",  # filepath
        1,   # hdu
        nothing,  # energy_units
        Dict{String,Vector}(),  # extra_columns
        Dict{String,Any}()  # headers
    )
    EventList(times, energies, dummy_meta)
end

function Base.show(io::IO, ::MIME"text/plain", ev::EventList)
    print(io, "EventList with $(length(ev.times)) times")
    if !isnothing(ev.energies)
        print(io, " and energies")
    end
    println(io)
end

# ============================================================================
# Interface Methods
# ============================================================================

Base.length(ev::EventList) = length(ev.times)
Base.size(ev::EventList) = (length(ev),)

# Accessor functions
times(ev::EventList) = ev.times
energies(ev::EventList) = ev.energies
has_energies(ev::EventList) = !isnothing(ev.energies)

# ============================================================================
# Filtering Functions (Composable and In-Place)
# ============================================================================

"""
    filter_time!(f, ev::EventList)

Filter all columns of the eventlist based on a predicate `f` applied to the
times. Modifies the EventList in-place.

# Example

```julia
# Filter only positive times
filter_time!(t -> t > 0, ev)

# Filter times greater than some minimum
filter_time!(>(min_time), ev)
```

See also [`filter_energy!`](@ref).
"""
filter_time!(f, ev::EventList) = filter_on!(f, ev.times, ev)

"""
    filter_energy!(f, ev::EventList)

Filter all columns of the eventlist based on a predicate `f` applied to the
energies. Modifies the EventList in-place.

# Example

```julia
# Filter energies less than 10 keV
filter_energy!(e -> e < 10.0, ev)

# With function composition
filter_energy!(<(10.0), ev)
```

See also [`filter_time!`](@ref).
"""
function filter_energy!(f, ev::EventList)
    @assert !isnothing(ev.energies) "No energies present in the EventList."
    filter_on!(f, ev.energies, ev)
end

"""
    filter_on!(f, src_col::AbstractVector, ev::EventList)

Internal function to filter EventList based on predicate applied to source column.
Uses efficient in-place filtering adapted from Base.filter! implementation.
"""
function filter_on!(f, src_col::AbstractVector, ev::EventList)
    @assert size(src_col) == size(ev.times) "Source column size must match times size"

    # Modified from Base.filter! implementation for multiple arrays
    j = firstindex(ev.times)

    for i in eachindex(ev.times)
        predicate = f(src_col[i])::Bool
        
        if predicate
            ev.times[j] = ev.times[i]

            if !isnothing(ev.energies)
                ev.energies[j] = ev.energies[i]
            end

            for (_, col) in ev.meta.extra_columns
                col[j] = col[i]
            end

            j = nextind(ev.times, j)
        end
    end

    # Resize all arrays to new length
    if j <= lastindex(ev.times)
        new_length = j - 1
        resize!(ev.times, new_length)

        if !isnothing(ev.energies)
            resize!(ev.energies, new_length)
        end

        for (_, col) in ev.meta.extra_columns
            resize!(col, new_length)
        end
    end

    ev
end

# ============================================================================
# Non-mutating Filter Functions
# ============================================================================

"""
    filter_time(f, ev::EventList)

Return a new EventList with events filtered by predicate `f` applied to times.
"""
function filter_time(f, ev::EventList)
    # Create a copy and filter in-place
    new_ev = deepcopy(ev)
    filter_time!(f, new_ev)
end

"""
    filter_energy(f, ev::EventList)

Return a new EventList with events filtered by predicate `f` applied to energies.
"""
function filter_energy(f, ev::EventList)
    # Create a copy and filter in-place
    new_ev = deepcopy(ev)
    filter_energy!(f, new_ev)
end

# ============================================================================
# File Reading Functions
# ============================================================================

"""
    colnames(file::AbstractString; hdu = 2)

Return a vector of all column names of `file`, reading from the specified HDU.
"""
function colnames(file::AbstractString; hdu = 2)
    FITS(file) do f
        selected_hdu = f[hdu]
        FITSIO.colnames(selected_hdu)
    end
end

"""
    read_energy_column(hdu; energy_alternatives = ["ENERGY", "PI", "PHA"], T = Float64)

Attempt to read the energy column of an HDU from a list of alternative names.
Returns `(column_name, data)` if successful, `(nothing, nothing)` if no column found.

This function is separated for:
- Simplified logic in main reading function
- Independent testing capability
- Type stability with explicit return types
"""
function read_energy_column(
    hdu; 
    energy_alternatives::Vector{String} = ["ENERGY", "PI", "PHA"], 
    T::Type = Float64
)::Tuple{Union{Nothing,String}, Union{Nothing,Vector{T}}}
    
    all_cols = uppercase.(FITSIO.colnames(hdu))
    
    for col_name in energy_alternatives
        if uppercase(col_name) in all_cols
            try
                data = read(hdu, col_name)
                return col_name, convert(Vector{T}, data)
            catch
                # If this column exists but can't be read, try the next one
                continue
            end
        end
    end
    
    return nothing, nothing
end

"""
    readevents(path; kwargs...)

Read an [`EventList`](@ref) from a FITS file. Will attempt to read an energy
column if one exists.

# Keyword arguments and defaults:
- `hdu::Int = 2`: which HDU unit to read
- `T::Type = Float64`: the type to cast the time and energy columns to
- `sort::Bool = false`: whether to sort by time if not already sorted
- `extra_columns::Vector{String} = []`: extra columns to read from the same HDU
- `mission::Union{String,Nothing} = nothing`: mission name for mission-specific handling
- `energy_alternatives::Vector{String} = ["ENERGY", "PI", "PHA"]`: energy column alternatives

# Type Stability
This function is designed to be type-stable with proper type annotations
on return values from FITS reading operations.
"""
function readevents(
    path::AbstractString;
    hdu::Int = 2,
    T::Type = Float64,
    sort::Bool = false,
    extra_columns::Vector{String} = String[],
    mission::Union{String,Nothing} = nothing,
    energy_alternatives::Vector{String} = ["ENERGY", "PI", "PHA"],
    kwargs...
)::EventList{Vector{T}, FITSMetadata{FITSIO.FITSHeader}}
    
    # Get mission-specific energy alternatives if mission is specified
    if !isnothing(mission)
        mission_support = get_mission_support(mission)
        energy_alternatives = mission_support.energy_alternatives
    end
    
    # Read data from FITS file with type-stable operations
    time::Vector{T}, energy::Union{Nothing,Vector{T}}, energy_col::Union{Nothing,String}, 
    header::FITSIO.FITSHeader, extra_data::Dict{String,Vector} = FITS(path, "r") do f
        
        selected_hdu = f[hdu]
        
        # Read header (type-stable)
        header = read_header(selected_hdu)
        
        # Read time column (type-stable conversion)
        time = convert(Vector{T}, read(selected_hdu, "TIME"))
        
        # Read energy column using separated function with mission-specific alternatives
        energy_column, energy = read_energy_column(
            selected_hdu; 
            T = T, 
            energy_alternatives = energy_alternatives
        )
        
        # Read extra columns
        extra_data = Dict{String,Vector}()
        for col_name in extra_columns
            extra_data[col_name] = read(selected_hdu, col_name)
        end
        
        (time, energy, energy_column, header, extra_data)
    end
    
    # Apply mission-specific calibration if needed
    if !isnothing(mission) && !isnothing(energy)
        mission_support = get_mission_support(mission)
        # Only apply calibration if we read PI or PHA columns (not ENERGY)
        if !isnothing(energy_col) && uppercase(energy_col) in ["PI", "PHA"]
            energy = apply_calibration(mission_support, energy)
        end
    end
    
    # Validate energy-time consistency
    if !isnothing(energy)
        @assert size(time) == size(energy) "Time and energy do not match sizes ($(size(time)) != $(size(energy)))"
    end
    
    # Handle sorting if requested
    if !issorted(time)
        if sort
            # Efficient sorting of multiple arrays
            sort_indices = sortperm(time)
            time = time[sort_indices]
            
            if !isnothing(energy)
                energy = energy[sort_indices]
            end
            
            # Sort extra columns
            for (col_name, col_data) in extra_data
                extra_data[col_name] = col_data[sort_indices]
            end
        else
            @assert false "Times are not sorted (pass `sort = true` to force sorting)"
        end
    end
    
    # Create metadata with mission-specific energy units
    energy_units = if !isnothing(mission) && !isnothing(energy_col)
        if uppercase(energy_col) in ["PI", "PHA"]
            "keV"  # After calibration
        else
            energy_col  # Original units
        end
    else
        energy_col
    end
    
    # Create metadata
    meta = FITSMetadata(path, hdu, energy_units, extra_data, header)
    
    # Return type-stable EventList
    EventList(time, energy, meta)
end

# ============================================================================
# GTI Support (Placeholder for Future Implementation)
# ============================================================================

"""
    read_gti(path::AbstractString; hdu::Int = 3)

Read Good Time Intervals from a FITS file. Returns (start_times, stop_times).
This is a placeholder for future GTI implementation.
"""
function read_gti(path::AbstractString; hdu::Int = 3)
    # TODO: Implement GTI reading
    # GTIs are typically in a separate HDU with START and STOP columns
    # This should be added in a separate PR as mentioned in the comments
    error("GTI reading not yet implemented")
end

"""
    filter_gti!(ev::EventList, gti_start::Vector, gti_stop::Vector)

Filter events to only include those within Good Time Intervals.
This is a placeholder for future GTI implementation.
"""
function filter_gti!(ev::EventList, gti_start::Vector, gti_stop::Vector)
    # TODO: Implement GTI filtering
    # This would use the composable filter_time! function
    error("GTI filtering not yet implemented")
end

# ============================================================================
# Utility Functions
# ============================================================================

"""
    summary(ev::EventList)

Provide a summary of the EventList contents.
"""
function Base.summary(ev::EventList)
    n_events = length(ev)
    time_span = isempty(ev.times) ? 0.0 : maximum(ev.times) - minimum(ev.times)
    
    summary_str = "EventList: $n_events events over $(time_span) time units"
    
    if has_energies(ev)
        energy_range = extrema(ev.energies)
        summary_str *= ", energies: $(energy_range[1]) - $(energy_range[2])"
        if !isnothing(ev.meta.energy_units)
            summary_str *= " ($(ev.meta.energy_units))"
        end
    end
    
    if !isempty(ev.meta.extra_columns)
        summary_str *= ", $(length(ev.meta.extra_columns)) extra columns"
    end
    
    return summary_str
end

# ============================================================================
# Exports
# ============================================================================

export EventList, FITSMetadata
export readevents, colnames, read_energy_column
export filter_time!, filter_energy!, filter_time, filter_energy
export times, energies, has_energies
export read_gti, filter_gti!  # Placeholders for future GTI support
# ============================================================================
# LIGHT CURVE STRUCTURES AND FUNCTIONS
# ============================================================================
"""
Abstract type for all light curve implementations.
"""
abstract type AbstractLightCurve{T} end

"""
    EventProperty{T}

A structure to hold additional event properties beyond time and energy.
"""
struct EventProperty{T}
    name::Symbol
    values::Vector{T}
    unit::String
end

"""
    LightCurveMetadata

A structure containing metadata for light curves.
"""
struct LightCurveMetadata
    telescope::String
    instrument::String
    object::String
    mjdref::Float64
    time_range::Tuple{Float64,Float64}
    bin_size::Float64
    headers::Vector{Dict{String,Any}}
    extra::Dict{String,Any}
end

"""
    LightCurve{T} <: AbstractLightCurve{T}

A structure representing a binned time series with additional properties.
"""
struct LightCurve{T} <: AbstractLightCurve{T}
    timebins::Vector{T}
    bin_edges::Vector{T}
    counts::Vector{Int}
    count_error::Vector{T}
    exposure::Vector{T}
    properties::Vector{EventProperty}
    metadata::LightCurveMetadata
    err_method::Symbol
end

"""
    calculate_errors(counts::Vector{Int}, method::Symbol, exposure::Vector{T}; 
                    gaussian_errors::Union{Nothing,Vector{T}}=nothing) where T

Calculate statistical uncertainties for count data using vectorized operations.
"""
function calculate_errors(counts::Vector{Int}, method::Symbol, exposure::Vector{T}; 
                         gaussian_errors::Union{Nothing,Vector{T}}=nothing) where T
    if method === :poisson
        # Vectorized Poisson errors: σ = sqrt(N), use sqrt(N + 1) when N = 0
        return convert.(T, @. sqrt(max(counts, 1)))
    elseif method === :gaussian
        if isnothing(gaussian_errors)
            throw(ArgumentError("Gaussian errors must be provided by user when using :gaussian method"))
        end
        if length(gaussian_errors) != length(counts)
            throw(ArgumentError("Length of gaussian_errors must match length of counts"))
        end
        return gaussian_errors
    else
        throw(ArgumentError("Unsupported error method: $method. Use :poisson or :gaussian"))
    end
end

"""
    validate_lightcurve_inputs(eventlist, binsize, err_method, gaussian_errors)

Validate all inputs for light curve creation before processing.
"""
function validate_lightcurve_inputs(eventlist, binsize, err_method, gaussian_errors)
    # Check event list
    if isempty(eventlist.times)
        throw(ArgumentError("Event list is empty"))
    end
    
    # Check bin size
    if binsize <= 0
        throw(ArgumentError("Bin size must be positive"))
    end
    
    # Check error method
    if !(err_method in [:poisson, :gaussian])
        throw(ArgumentError("Unsupported error method: $err_method. Use :poisson or :gaussian"))
    end
    
    # Check Gaussian errors if needed
    if err_method === :gaussian
        if isnothing(gaussian_errors)
            throw(ArgumentError("Gaussian errors must be provided when using :gaussian method"))
        end
        # Note: Length validation will happen after filtering, not here
    end
end

"""
    apply_event_filters(times::Vector{T}, energies::Union{Nothing,Vector{T}}, 
                       tstart::Union{Nothing,Real}, tstop::Union{Nothing,Real},
                       energy_filter::Union{Nothing,Tuple{Real,Real}}) where T

Apply time and energy filters to event data.
Returns filtered times and energies.
"""
function apply_event_filters(times::Vector{T}, energies::Union{Nothing,Vector{T}}, 
                            tstart::Union{Nothing,Real}, tstop::Union{Nothing,Real},
                            energy_filter::Union{Nothing,Tuple{Real,Real}}) where T
    
    filtered_times = times
    filtered_energies = energies
    
    # Apply energy filter first if specified
    if !isnothing(energy_filter) && !isnothing(energies)
        emin, emax = energy_filter
        energy_mask = @. (energies >= emin) & (energies < emax)
        filtered_times = times[energy_mask]
        filtered_energies = energies[energy_mask]
        
        if isempty(filtered_times)
            throw(ArgumentError("No events remain after energy filtering"))
        end
        @info "Applied energy filter [$emin, $emax) keV: $(length(filtered_times)) events remain"
    end
    
    # Determine time range
    start_time = isnothing(tstart) ? minimum(filtered_times) : convert(T, tstart)
    stop_time = isnothing(tstop) ? maximum(filtered_times) : convert(T, tstop)
    
    # Apply time filter if needed
    if start_time != minimum(filtered_times) || stop_time != maximum(filtered_times)
        time_mask = @. (filtered_times >= start_time) & (filtered_times <= stop_time)
        filtered_times = filtered_times[time_mask]
        if !isnothing(filtered_energies)
            filtered_energies = filtered_energies[time_mask]
        end
        
        if isempty(filtered_times)
            throw(ArgumentError("No events remain after time filtering"))
        end
        @info "Applied time filter [$start_time, $stop_time]: $(length(filtered_times)) events remain"
    end
    
    return filtered_times, filtered_energies, start_time, stop_time
end

"""
    create_time_bins(start_time::T, stop_time::T, binsize::T) where T

Create time bin edges and centers for the light curve.
"""
function create_time_bins(start_time::T, stop_time::T, binsize::T) where T
    # Ensure we cover the full range including the endpoint
    start_bin = floor(start_time / binsize) * binsize
    
    # Calculate number of bins to ensure we cover stop_time
    time_span = stop_time - start_bin
    num_bins = max(1, ceil(Int, time_span / binsize))
    
    # Adjust if the calculated end would be less than stop_time
    while start_bin + num_bins * binsize < stop_time
        num_bins += 1
    end
    
    # Create bin edges and centers efficiently
    edges = [start_bin + i * binsize for i in 0:num_bins]
    centers = [start_bin + (i + 0.5) * binsize for i in 0:(num_bins-1)]
    
    return edges, centers
end

"""
    bin_events(times::Vector{T}, bin_edges::Vector{T}) where T

Bin event times into histogram counts.
"""
function bin_events(times::Vector{T}, bin_edges::Vector{T}) where T
    # Use StatsBase for fast, memory-efficient binning
    hist = fit(Histogram, times, bin_edges)
    return Vector{Int}(hist.weights)
end

"""
    calculate_additional_properties(times::Vector{T}, energies::Union{Nothing,Vector{U}}, 
                                   bin_edges::Vector{T}, bin_centers::Vector{T}) where {T,U}

Calculate additional properties like mean energy per bin.
Fixed to handle type mismatches between time and energy vectors.
"""
function calculate_additional_properties(times::Vector{T}, energies::Union{Nothing,Vector{U}}, 
                                        bin_edges::Vector{T}, bin_centers::Vector{T}) where {T,U}
    properties = Vector{EventProperty}()
    
    # Calculate mean energy per bin if available
    if !isnothing(energies) && !isempty(energies) && length(bin_centers) > 0
        start_bin = bin_edges[1]
        
        # Handle case where there's only one bin center
        if length(bin_centers) == 1
            binsize = length(bin_edges) > 1 ? bin_edges[2] - bin_edges[1] : T(1)
        else
            binsize = bin_centers[2] - bin_centers[1]  # Assuming uniform bins
        end
        
        # Use efficient binning for energies
        energy_sums = zeros(T, length(bin_centers))
        energy_counts = zeros(Int, length(bin_centers))
        
        # Vectorized binning for energies
        for (t, e) in zip(times, energies)
            bin_idx = floor(Int, (t - start_bin) / binsize) + 1
            if 1 ≤ bin_idx ≤ length(bin_centers)
                energy_sums[bin_idx] += T(e)  # Convert energy to time type
                energy_counts[bin_idx] += 1
            end
        end
        
        # Calculate mean energies using vectorized operations
        mean_energy = @. ifelse(energy_counts > 0, energy_sums / energy_counts, zero(T))
        push!(properties, EventProperty{T}(:mean_energy, mean_energy, "keV"))
    end
    
    return properties
end

"""
    extract_metadata(eventlist, start_time, stop_time, binsize, filtered_times, energy_filter)

Extract and create metadata for the light curve.
"""
function extract_metadata(eventlist, start_time, stop_time, binsize, filtered_times, energy_filter)
    first_header = isempty(eventlist.metadata.headers) ? Dict{String,Any}() : eventlist.metadata.headers[1]
    
    return LightCurveMetadata(
        get(first_header, "TELESCOP", ""),
        get(first_header, "INSTRUME", ""),
        get(first_header, "OBJECT", ""),
        get(first_header, "MJDREF", 0.0),
        (Float64(start_time), Float64(stop_time)),
        Float64(binsize),
        eventlist.metadata.headers,
        Dict{String,Any}(
            "filtered_nevents" => length(filtered_times),
            "total_nevents" => length(eventlist.times),
            "energy_filter" => energy_filter
        )
    )
end

"""
    create_lightcurve(
        eventlist::EventList{T}, 
        binsize::Real;
        err_method::Symbol=:poisson,
        gaussian_errors::Union{Nothing,Vector{T}}=nothing,
        tstart::Union{Nothing,Real}=nothing,
        tstop::Union{Nothing,Real}=nothing,
        energy_filter::Union{Nothing,Tuple{Real,Real}}=nothing,
        event_filter::Union{Nothing,Function}=nothing
    ) where T

Create a light curve from an event list with enhanced performance and filtering.

# Arguments
- `eventlist`: The input event list
- `binsize`: Time bin size
- `err_method`: Error calculation method (:poisson or :gaussian)
- `gaussian_errors`: User-provided Gaussian errors (required if err_method=:gaussian)
- `tstart`, `tstop`: Time range limits
- `energy_filter`: Energy range as (emin, emax) tuple
- `event_filter`: Optional function to filter events, should return boolean mask
"""
function create_lightcurve(
    eventlist::EventList{T}, 
    binsize::Real;
    err_method::Symbol=:poisson,
    gaussian_errors::Union{Nothing,Vector{T}}=nothing,
    tstart::Union{Nothing,Real}=nothing,
    tstop::Union{Nothing,Real}=nothing,
    energy_filter::Union{Nothing,Tuple{Real,Real}}=nothing,
    event_filter::Union{Nothing,Function}=nothing
) where T
    
    # Validate all inputs first
    validate_lightcurve_inputs(eventlist, binsize, err_method, gaussian_errors)
    
    binsize_t = convert(T, binsize)
    
    # Get initial data references
    times = eventlist.times
    energies = eventlist.energies
    
    # Apply custom event filter if provided
    if !isnothing(event_filter)
        filter_mask = event_filter(eventlist)
        if !isa(filter_mask, AbstractVector{Bool})
            throw(ArgumentError("Event filter function must return a boolean vector"))
        end
        if length(filter_mask) != length(times)
            throw(ArgumentError("Event filter mask length must match number of events"))
        end
        
        times = times[filter_mask]
        if !isnothing(energies)
            energies = energies[filter_mask]
        end
        
        if isempty(times)
            throw(ArgumentError("No events remain after custom filtering"))
        end
        @info "Applied custom filter: $(length(times)) events remain"
    end
    
    # Apply standard filters
    filtered_times, filtered_energies, start_time, stop_time = apply_event_filters(
        times, energies, tstart, tstop, energy_filter
    )
    
    # Create time bins
    bin_edges, bin_centers = create_time_bins(start_time, stop_time, binsize_t)
    
    # Bin the events
    counts = bin_events(filtered_times, bin_edges)
    
    @info "Created light curve: $(length(bin_centers)) bins, bin size = $(binsize_t) s"
    
    # Now validate gaussian_errors length if needed
    if err_method === :gaussian && !isnothing(gaussian_errors)
        if length(gaussian_errors) != length(counts)
            throw(ArgumentError("Length of gaussian_errors ($(length(gaussian_errors))) must match number of bins ($(length(counts)))"))
        end
    end
    
    # Calculate exposures and errors
    exposure = fill(binsize_t, length(bin_centers))
    errors = calculate_errors(counts, err_method, exposure; gaussian_errors=gaussian_errors)
    
    # Calculate additional properties
    properties = calculate_additional_properties(filtered_times, filtered_energies, bin_edges, bin_centers)
    
    # Extract metadata
    metadata = extract_metadata(eventlist, start_time, stop_time, binsize_t, filtered_times, energy_filter)
    
    return LightCurve{T}(
        bin_centers,
        bin_edges,
        counts,
        errors,
        exposure,
        properties,
        metadata,
        err_method
    )
end

"""
    rebin(lc::LightCurve{T}, new_binsize::Real; 
          gaussian_errors::Union{Nothing,Vector{T}}=nothing) where T

Rebin a light curve to a new time resolution with enhanced performance.
"""
function rebin(lc::LightCurve{T}, new_binsize::Real; 
               gaussian_errors::Union{Nothing,Vector{T}}=nothing) where T
    if new_binsize <= lc.metadata.bin_size
        throw(ArgumentError("New bin size must be larger than current bin size"))
    end
    
    old_binsize = T(lc.metadata.bin_size)
    new_binsize_t = convert(T, new_binsize)
    
    # Create new bin edges using the same approach as in create_lightcurve
    start_time = T(lc.metadata.time_range[1])
    stop_time = T(lc.metadata.time_range[2])
    
    # Calculate bin edges using efficient algorithm
    start_bin = floor(start_time / new_binsize_t) * new_binsize_t
    time_span = stop_time - start_bin
    num_bins = max(1, ceil(Int, time_span / new_binsize_t))
    
    # Ensure we cover the full range
    while start_bin + num_bins * new_binsize_t < stop_time
        num_bins += 1
    end
    
    new_edges = [start_bin + i * new_binsize_t for i in 0:num_bins]
    new_centers = [start_bin + (i + 0.5) * new_binsize_t for i in 0:(num_bins-1)]
    
    # Rebin counts using vectorized operations where possible
    new_counts = zeros(Int, length(new_centers))
    
    for (i, time) in enumerate(lc.timebins)
        if lc.counts[i] > 0  # Only process bins with counts
            bin_idx = floor(Int, (time - start_bin) / new_binsize_t) + 1
            if 1 ≤ bin_idx ≤ length(new_counts)
                new_counts[bin_idx] += lc.counts[i]
            end
        end
    end
    
    # Calculate new exposures and errors
    new_exposure = fill(new_binsize_t, length(new_centers))
    
    # Handle error propagation based on original method
    if lc.err_method === :gaussian && isnothing(gaussian_errors)
        throw(ArgumentError("Gaussian errors must be provided when rebinning a light curve with Gaussian errors"))
    end
    
    new_errors = calculate_errors(new_counts, lc.err_method, new_exposure; gaussian_errors=gaussian_errors)
    
    # Rebin properties using weighted averaging
    new_properties = Vector{EventProperty}()
    for prop in lc.properties
        new_values = zeros(T, length(new_centers))
        counts = zeros(Int, length(new_centers))
        
        for (i, val) in enumerate(prop.values)
            if lc.counts[i] > 0  # Only process bins with counts
                bin_idx = floor(Int, (lc.timebins[i] - start_bin) / new_binsize_t) + 1
                if 1 ≤ bin_idx ≤ length(new_values)
                    new_values[bin_idx] += val * lc.counts[i]
                    counts[bin_idx] += lc.counts[i]
                end
            end
        end
        
        # Calculate weighted average using vectorized operations
        new_values = @. ifelse(counts > 0, new_values / counts, zero(T))
        
        push!(new_properties, EventProperty(prop.name, new_values, prop.unit))
    end
    
    # Update metadata
    new_metadata = LightCurveMetadata(
        lc.metadata.telescope,
        lc.metadata.instrument,
        lc.metadata.object,
        lc.metadata.mjdref,
        lc.metadata.time_range,
        Float64(new_binsize_t),
        lc.metadata.headers,
        merge(
            lc.metadata.extra,
            Dict{String,Any}("original_binsize" => Float64(old_binsize))
        )
    )
    
    return LightCurve{T}(
        new_centers,
        new_edges,
        new_counts,
        new_errors,
        new_exposure,
        new_properties,
        new_metadata,
        lc.err_method
    )
end

# Basic array interface methods
Base.length(lc::LightCurve) = length(lc.counts)
Base.size(lc::LightCurve) = (length(lc),)
Base.getindex(lc::LightCurve, i) = (lc.timebins[i], lc.counts[i])
@recipe function f(el::EventList, bin_size::Real=1.0; 
                  tstart=nothing, tstop=nothing, tseg=nothing, 
                  show_errors=true, show_gaps=false, gap_threshold=10.0, 
                  axis_limits=nothing, max_events_for_gaps=1_000_000,
                  energy_filter=nothing, normalize=false)
    
    # Input validation
    if isempty(el.times)
        error("EventList is empty - no events to plot")
    end
    
    if bin_size <= 0
        error("bin_size must be positive")
    end
    
    # Get and filter times efficiently  
    times = el.times
    
    # Apply energy filter first if specified (reduces data early)
    if !isnothing(energy_filter) && !isnothing(el.energies)
        emin, emax = energy_filter
        energy_mask = @. (el.energies >= emin) & (el.energies < emax)
        times = times[energy_mask]
        if isempty(times)
            error("No events remain after energy filtering")
        end
    end
    
    # Determine time range efficiently
    t_min = isnothing(tstart) ? minimum(times) : Float64(tstart)
    t_max = if !isnothing(tstop)
        Float64(tstop)
    elseif !isnothing(tseg)
        t_min + Float64(tseg)
    else
        maximum(times)
    end
    
    # Apply time filter using vectorized operations
    if t_min != minimum(times) || t_max != maximum(times)
        time_mask = @. (times >= t_min) & (times <= t_max)
        times = times[time_mask]
    end
    
    if isempty(times)
        error("No events in specified time range")
    end
    
    # Create efficient binning - use the same approach as debug function
    bin_size_t = Float64(bin_size)
    start_edge = floor(t_min / bin_size_t) * bin_size_t
    end_edge = ceil(t_max / bin_size_t) * bin_size_t
    edges = start_edge:bin_size_t:end_edge
    
    # Use StatsBase.fit for fast histogram computation
    hist = fit(Histogram, times, collect(edges))
    counts = Vector{Float64}(hist.weights)  # Convert Int64 weights to Float64
    bin_centers = collect(edges[1:end-1] .+ bin_size_t / 2)
    
    # Normalize if requested
    ylabel_text = if normalize
        counts = counts ./ bin_size_t
        "Count Rate (counts/s)"
    else
        "Counts"
    end
    
    # Set plot attributes
    xlabel := "Time (s)"
    ylabel := ylabel_text
    
    # Apply axis limits if specified
    if !isnothing(axis_limits)
        xlims := axis_limits[1]
        ylims := axis_limits[2]
    end
    
    # Main series - return the data directly
    @series begin
        seriestype := :line
        linewidth := 2
        label := "Light Curve"
        bin_centers, counts
    end
    
    # Add error bars if requested
    if show_errors
        errors = sqrt.(max.(counts, 1.0))
        if normalize
            errors = errors ./ bin_size_t
        end
        
        @series begin
            seriestype := :scatter
            yerror := errors
            markersize := 0
            linewidth := 1
            color := :black
            alpha := 0.7
            label := ""
            bin_centers, counts
        end
    end
    
    # Add gap detection for smaller datasets
    if show_gaps && length(bin_centers) <= max_events_for_gaps
        gap_indices = find_gaps(bin_centers, counts, gap_threshold * bin_size_t)
        
        if !isempty(gap_indices)
            @series begin
                seriestype := :vline
                color := :red
                linestyle := :dash
                alpha := 0.5
                linewidth := 1
                label := "Data Gaps"
                bin_centers[gap_indices]
            end
        end
    end
end

# Recipe for plotting LightCurve objects directly
@recipe function f(lc::LightCurve{T}; 
                  show_errors=true, show_gaps=false, gap_threshold=10.0,
                  normalize=false, axis_limits=nothing) where T
    
    times = lc.timebins
    counts = Vector{Float64}(lc.counts)  # Proper vector conversion
    
    # Normalize if requested
    if normalize
        counts = counts ./ lc.metadata.bin_size
        ylabel_text = "Count Rate (counts/s)"
        errors = Vector{Float64}(lc.count_error) ./ lc.metadata.bin_size
    else
        ylabel_text = "Counts"
        errors = Vector{Float64}(lc.count_error)
    end
    
    # Main plot
    @series begin
        seriestype := :line
        linewidth := 2
        label := "Light Curve"
        xlabel := "Time (s)"
        ylabel := ylabel_text
        
        if !isnothing(axis_limits)
            xlims := axis_limits[1]
            ylims := axis_limits[2]
        end
        
        times, counts
    end
    
    # Error bars
    if show_errors
        @series begin
            seriestype := :scatter
            yerror := errors
            markersize := 0
            linewidth := 1
            color := :black
            alpha := 0.7
            label := ""
            
            times, counts
        end
    end
    
    # Gap detection
    if show_gaps
        gap_indices = find_gaps(times, lc.counts, gap_threshold * lc.metadata.bin_size)
        
        if !isnothing(gap_indices) && !isempty(gap_indices)
            @series begin
                seriestype := :vline
                color := :red
                linestyle := :dash
                alpha := 0.5
                linewidth := 1
                label := "Data Gaps"
                
                times[gap_indices]
            end
        end
    end
end

# Helper function for gap detection (you'll need to implement this)
function find_gaps(times::Vector, counts::Vector, threshold::Real)
    # Simple gap detection - find large time differences
    gaps = Int[]
    if length(times) < 2
        return gaps
    end
    
    dt = diff(times)
    gap_mask = dt .> threshold
    
    for (i, is_gap) in enumerate(gap_mask)
        if is_gap
            push!(gaps, i+1)  # Index of the bin after the gap
        end
    end
    
    return gaps
end
using Test
using FITSIO
using Logging

"""
Dictionary of simple conversion functions for different missions.

This dictionary provides PI (Pulse Invariant) to energy conversion functions
for various X-ray astronomy missions. Each function takes a PI channel value
and returns the corresponding energy in keV.

Supported missions:
- NuSTAR: Nuclear Spectroscopic Telescope Array
- XMM: X-ray Multi-Mirror Mission  
- NICER: Neutron star Interior Composition Explorer
- IXPE: Imaging X-ray Polarimetry Explorer
- AXAF/Chandra: Advanced X-ray Astrophysics Facility
- XTE/RXTE: Rossi X-ray Timing Explorer
"""
const SIMPLE_CALIBRATION_FUNCS = Dict{String, Function}(
    "nustar" => (pi) -> pi * 0.04 + 1.62,
    "xmm" => (pi) -> pi * 0.001,
    "nicer" => (pi) -> pi * 0.01,
    "ixpe" => (pi) -> pi / 375 * 15,
    "axaf" => (pi) -> (pi - 1) * 14.6e-3,  # Chandra/AXAF
    "chandra" => (pi) -> (pi - 1) * 14.6e-3,  # Explicit chandra entry
    "xte" => (pi) -> pi * 0.025  # RXTE/XTE
)

"""
Abstract type for mission-specific calibration and interpretation.

This serves as the base type for all mission support implementations,
allowing for extensibility and type safety in mission-specific operations.
"""
abstract type AbstractMissionSupport end

"""
    MissionSupport{T} <: AbstractMissionSupport

Structure containing mission-specific calibration and interpretation information.

This structure encapsulates all the necessary information for handling
data from a specific X-ray astronomy mission, including calibration
functions, energy column alternatives, and GTI extension preferences.

# Fields
- `name::String`: Mission name (normalized to lowercase)
- `instrument::Union{String, Nothing}`: Instrument identifier
- `epoch::Union{T, Nothing}`: Observation epoch in MJD (for time-dependent calibrations)
- `calibration_func::Function`: PI to energy conversion function
- `interpretation_func::Union{Function, Nothing}`: Mission-specific FITS interpretation function
- `energy_alternatives::Vector{String}`: Preferred energy column names in order of preference
- `gti_extensions::Vector{String}`: GTI extension names in order of preference

# Type Parameters
- `T`: Type of the epoch parameter (typically Float64)
"""
struct MissionSupport{T} <: AbstractMissionSupport
    name::String
    instrument::Union{String, Nothing}
    epoch::Union{T, Nothing}
    calibration_func::Function
    interpretation_func::Union{Function, Nothing}
    energy_alternatives::Vector{String}
    gti_extensions::Vector{String}
end

"""
    get_mission_support(mission::String, instrument=nothing, epoch=nothing) -> MissionSupport

Create mission support object with mission-specific parameters.

This function creates a MissionSupport object containing all the necessary
information for processing data from a specified X-ray astronomy mission.
It handles mission aliases (e.g., Chandra/AXAF) and provides appropriate
defaults for each mission.

# Arguments
- `mission::String`: Mission name (case-insensitive)
- `instrument::Union{String, Nothing}=nothing`: Instrument identifier
- `epoch::Union{Float64, Nothing}=nothing`: Observation epoch in MJD

# Returns
- `MissionSupport{Float64}`: Mission support object

# Throws
- `ArgumentError`: If mission name is empty

# Examples
```julia
# Basic usage
ms = get_mission_support("nustar")

# With instrument specification
ms = get_mission_support("nustar", "FPM_A")

# With epoch for time-dependent calibrations
ms = get_mission_support("xte", "PCA", 50000.0)
```
"""
function get_mission_support(mission::String, 
                           instrument::Union{String, Nothing}=nothing,
                           epoch::Union{Float64, Nothing}=nothing)
    
    # Check for empty mission string
    if isempty(mission)
        throw(ArgumentError("Mission name cannot be empty"))
    end
    
    mission_lower = lowercase(mission)
    
    # Handle chandra/axaf aliases - normalize to chandra
    if mission_lower in ["chandra", "axaf"]
        mission_lower = "chandra"
    end
    
    calib_func = if haskey(SIMPLE_CALIBRATION_FUNCS, mission_lower)
        SIMPLE_CALIBRATION_FUNCS[mission_lower]
    else
        @warn "Mission $mission not recognized, using identity function"
        identity
    end
    
    # Mission-specific energy alternatives (order matters!)
    energy_alts = if mission_lower in ["chandra", "axaf"]
        ["ENERGY", "PI", "PHA"]  # Chandra usually has ENERGY column
    elseif mission_lower == "xte"
        ["PHA", "PI", "ENERGY"]
    elseif mission_lower == "nustar"
        ["PI", "ENERGY", "PHA"]
    else
        ["ENERGY", "PI", "PHA"]
    end
    
    # Mission-specific GTI extensions
    gti_exts = if mission_lower == "xmm"
        ["GTI", "GTI0", "STDGTI"]
    elseif mission_lower in ["chandra", "axaf"]
        ["GTI", "GTI0", "GTI1", "GTI2", "GTI3"]
    else
        ["GTI", "STDGTI"]
    end
    
    MissionSupport{Float64}(mission_lower, instrument, epoch, calib_func, nothing, energy_alts, gti_exts)
end

"""
    apply_calibration(mission_support::MissionSupport, pi_channels::AbstractArray) -> Vector{Float64}

Apply calibration function to PI channels.

Converts PI (Pulse Invariant) channel values to energies in keV using
the mission-specific calibration function stored in the MissionSupport object.

# Arguments
- `mission_support::MissionSupport`: Mission support object containing calibration function
- `pi_channels::AbstractArray{T}`: Array of PI channel values

# Returns
- `Vector{Float64}`: Array of energy values in keV

# Examples
```julia
ms = get_mission_support("nustar")
pi_values = [100, 500, 1000]
energies = apply_calibration(ms, pi_values)
```
"""
function apply_calibration(mission_support::MissionSupport, pi_channels::AbstractArray{T}) where T
    if isempty(pi_channels)
        return similar(pi_channels, Float64)
    end
    return mission_support.calibration_func.(pi_channels)
end

"""
    patch_mission_info(info::Dict{String,Any}, mission=nothing) -> Dict{String,Any}

Apply mission-specific patches to header information.

This function applies mission-specific modifications to FITS header information
to handle mission-specific quirks and conventions. It's based on the Python
implementation in Stingray's mission interpretation module.

# Arguments
- `info::Dict{String,Any}`: Dictionary containing header information
- `mission::Union{String,Nothing}=nothing`: Mission name

# Returns
- `Dict{String,Any}`: Patched header information dictionary

# Examples
```julia
info = Dict("gti" => "STDGTI", "ecol" => "PHA")
patched = patch_mission_info(info, "xmm")  # Adds GTI0 to gti field
```
"""
function patch_mission_info(info::Dict{String,Any}, mission::Union{String,Nothing}=nothing)
    if isnothing(mission)
        return info
    end
    
    mission_lower = lowercase(mission)
    patched_info = copy(info)
    
    # Normalize chandra/axaf
    if mission_lower in ["chandra", "axaf"]
        mission_lower = "chandra"
    end
    
    if mission_lower == "xmm" && haskey(patched_info, "gti")
        patched_info["gti"] = string(patched_info["gti"], ",GTI0")
    elseif mission_lower == "xte" && haskey(patched_info, "ecol")
        patched_info["ecol"] = "PHA"
        patched_info["ccol"] = "PCUID"
    elseif mission_lower == "chandra"
        # Chandra-specific patches
        if haskey(patched_info, "DETNAM")
            patched_info["detector"] = patched_info["DETNAM"]
        end
        # Add Chandra-specific time reference if needed
        if haskey(patched_info, "TIMESYS")
            patched_info["time_system"] = patched_info["TIMESYS"]
        end
    end
    
    return patched_info
end
function interpret_fits_data!(f::FITS, mission_support::MissionSupport)
    # Placeholder for mission-specific interpretation
    # This would contain mission-specific FITS handling logic
    return nothing
end