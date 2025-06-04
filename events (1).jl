using FITSIO
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